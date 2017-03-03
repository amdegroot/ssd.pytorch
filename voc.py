# voc.py
"""Adaptation of @fmassa's voc_dataset branch of torch-vision

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
"""
import os
import os.path
import sys


from config import VOCroot

import torch
import torch.utils.data as data
from PIL import Image, ImageDraw
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

CLASSES = ('__background__',  # always index 0
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


def _flip_box(boxes, width):
    boxes = boxes.clone()
    oldx1 = boxes[:, 0].clone()
    oldx2 = boxes[:, 2].clone()
    boxes[:, 0] = width - oldx2 - 1
    boxes[:, 2] = width - oldx1 - 1
    return boxes


class TransformVOCAnnotation(object):
    """
    Arguments:
        class_to_ind (dict): {<classname> : <classindex>}
        keep_difficult (bool): whether or not to keep difficult instances
            defualt: False
    """

    def __init__(self, class_to_ind, keep_difficult=False):
        self.keep_difficult = keep_difficult
        self.class_to_ind = class_to_ind

    def __call__(self, target):
        """
        Arguments:
            target (Annotation) : the target annotation to be made usable
        Returns:
            a dict with the following entries
                - boxes (LongTensor): tensor representation fo all bounding
                    boxes
                - gt_classes ([int]): array of class indices for each gt obj
                - im_dim ([int]): array containing [width, height,1]
        """
        boxes = []
        gt_classes = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bb = obj.find('bndbox')
            bndbox = map(int, [bb.find('xmin').text, bb.find('ymin').text,
                               bb.find('xmax').text, bb.find('ymax').text])

            boxes += [bndbox]
            gt_classes += [self.class_to_ind[name]]

        size = target.find('size')
        im_dim = [int(i) for i in [size.find('width').text,
                                   size.find('height').text, 1]]

        res = {
            'boxes': torch.LongTensor(boxes),
            'gt_classes': gt_classes,
            'im_dim': im_dim
        }
        return res


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (TransformVOCAnnotation): the transformation to be applied
            to the image
        target_transform (TransformVOCAnnotation): the transformation to be
            applied to the target
    """

    def __init__(self, root, image_set, transform=None, target_transform=None,
                 dataset_name='VOC2007'):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        self._annopath = os.path.join(
            self.root, dataset_name, 'Annotations', '%s.xml')
        self._imgpath = os.path.join(
            self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(
            self.root, dataset_name, 'ImageSets', 'Main', '%s.txt')

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()

        img = Image.open(self._imgpath % img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def show(self, index):
        '''Shows an image with it's bounding box overlaid

        Argument:
            index (int): index of img to show
        '''
        img, target = self.__getitem__(index)
        draw = ImageDraw.Draw(img)
        i = 0
        for obj in target.iter('object'):
            bb = obj.find('bndbox')
            bbcoord = [int(b.text) for b in bb.getchildren()]  # [x1,y1,x2,y2]
            draw.rectangle(bbcoord, outline=COLORS[i % len(COLORS)])
            draw.text(bbcoord[:2], obj.find('name').text,
                      fill=COLORS[(i + 3) % len(COLORS)])
            i += 1
        img.show()

if __name__ == '__main__':
    class_to_ind = dict(zip(CLASSES, range(len(CLASSES))))

    ds = VOCDetection(VOCroot, 'train',
                      target_transform=TransformVOCAnnotation(class_to_ind, False))
    print(len(ds))
    img, target = ds[0]
    print(target)
    # ds.show(1)
    # dss = VOCSegmentation(VOCroot, 'train')
    # img, target = ds[0]

    # img.show()
    # print(target_transform(target))
