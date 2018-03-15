import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import boto3
import tempfile


if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


# GLOBALS
CLASSES = ('person', 'basketball')
# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class AnnotationTransformBhjc(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False, ball_only=False):
        self.class_to_ind = class_to_ind or dict(
            zip(CLASSES, range(len(CLASSES))))
        self.keep_difficult = keep_difficult
        self.ball_only = ball_only

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            # move to next object if not a ball and we've decided on ball detection only
            if self.ball_only and 'ball' not in name:
                continue

            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class BhjcBballDataset(data.Dataset):

    """
    Dataset supporting the labelImg labeled images from our 2018-01-23 experiment
    at the Beverly Hills Jewish Community Center.

    __init__ Arguments:
        img_path (string): filepath to the folder containing images
        anno_path (string): filepath to the folder containing annotations
        transform (callable, optional): transformation to be performed on image
        target_transform (callable, optional): trasformation to be performed on annotations
        id_list = list of image ids
        dataset_name (string, optional): name of the dataset
            (default: 'bhjc')
    """
    #TODO: read image files from s3
    def __init__(self, img_path, anno_path, id_list, transform=None,
                 target_transform=None, dataset_name='bhjc', file_name_prfx='left_scene2_rot180_'):
        self.img_path = img_path
        self.anna_path = anno_path
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = anno_path
        self._imgpath = img_path
        self.ids = id_list
        self.file_name_prfx = file_name_prfx

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def get_img_targ_from_s3(self, img_id, s3_bucket='geniussports-computer-vision-data',
                             s3_path='internal-experiments/basketball/bhjc/20180123/'):

        im_path = s3_path + 'images/left_cam/' + self.file_name_prfx + img_id + '.png'
        anno_path = s3_path + 'labels/' + self.file_name_prfx + img_id + '.xml'  # xml file has no left_cam directory

        print('loading:', im_path)
        print('loading:', anno_path)

        s3 = boto3.resource('s3', region_name='us-west-2')
        bucket = s3.Bucket(s3_bucket)
        im_obj = bucket.Object(im_path)
        anno_obj = bucket.Object(anno_path)

        tmp = tempfile.NamedTemporaryFile()

        # dowload to temp file and read in
        with open(tmp.name, 'wb') as f:
            im_obj.download_fileobj(f)
        img = cv2.imread(tmp.name)

        with open(tmp.name, 'wb') as f:
            anno_obj.download_fileobj(f)
        target = ET.parse(tmp.name).getroot()

        return img, target

    def pull_item(self, index):
        img_id = self.ids[index]

        # anno_file = self._annopath + self.file_name_prfx + img_id + '.xml'
        # img_file = self._imgpath + self.file_name_prfx + img_id + '.png'

        # print(anno_file)
        # print(img_file)

        # target = ET.parse(anno_file).getroot()
        # img = cv2.imread(img_file)

        img, target = self.get_img_targ_from_s3(img_id)

        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width





