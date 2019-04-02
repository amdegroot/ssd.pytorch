"""VisDrone2018 Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

DRONE_CLASSES = (  #  1+11=12
    'pedestrian', 'person', 'bicycle', 'car',
    'van', 'truck', 'tricycle', 'awning-tricycle', 'bus',
    'motor', 'others')

# note: if you used our download scripts, this should be right
# make sure path does not contains substring 'annotations'
DRONE_ROOT = "/media/mk/本地磁盘/Datasets/UAV/VisDrone2018"


class DroneAnnotationTransform(object):
    """Transforms a VisDrone2018 annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VisDrone2018's 11 classes, 1 ignore)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(DRONE_CLASSES, range(len(DRONE_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, filename, frameidx, width, height):
        """
        Arguments:
            filename: anno文件的名字
            frameidx: 文件中对应的视频段的哪一帧，每行第一个数
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
            [x0, y0, x1, y1, category(int)]
            坐标进行过标准化，变成[0., 1.]的小数
        """

        # target: filename
        res = []
        with open(filename+'.txt', 'r') as f:
            lines = f.readlines()
            frameanno = []
            for itemline in lines:
                item = itemline.strip().split(',')  # 去换行符，逗号分割
                # print(item)
                
                if int(item[0]) == frameidx: # item[6] == 0?
                    bbox = [float(item[2])/width, float(item[3])/height, (float(item[2])+float(item[4]))/width, (float(item[3])+float(item[5]))/height, int(item[7])-1]
                    # 注意！！！
                    # 导致错误  RuntimeError: cuda runtime error (59) : device-side assert triggered at /opt/conda/conda-bld/pytorch_1544199946412/work/aten/src/THC/generated/../THCTensorMathCompareT.cuh:69
                    #          RuntimeError: copy_if failed to synchronize: device-side assert triggered
                    # 输入1-x转为0-(x-1)，box_utils.py中再补0，否则数组超限
                    # int(item[7])-1原因为统一成输入标注从0开始，再在layers/box_utils.py中转换为0非物体，1-x物体，item为从1-11的标注，bbox为0-10，和voc 0-19统一
                    # print('bbox'+str(bbox))
                    res += [bbox]
            # print('res.shape: ' + str(np.array(res).shape))
        
        return res






        # res = []
        # for obj in target.iter('object'):
        #     difficult = int(obj.find('difficult').text) == 1
        #     if not self.keep_difficult and difficult:
        #         continue
        #     name = obj.find('name').text.lower().strip()
        #     bbox = obj.find('bndbox')

        #     pts = ['xmin', 'ymin', 'xmax', 'ymax']
        #     bndbox = []
        #     for i, pt in enumerate(pts):
        #         cur_pt = int(bbox.find(pt).text) - 1
        #         # scale height or width
        #         cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
        #         bndbox.append(cur_pt)
        #     label_idx = self.class_to_ind[name]
        #     bndbox.append(label_idx)
        #     res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        #     # img_id = target.find('filename').text[:-4]

        # return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class DroneDetection(data.Dataset):
    """VisDrone2018 Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VisDrone2018 devkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VisDrone2018 2007')
    """

    def __init__(self, root,
                 transform=None, target_transform=DroneAnnotationTransform(),
                 dataset_name='VisDrone 2018', train=1):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        if train == 1:
            self._annopath = osp.join(self.root, 'VisDrone2018-VID-train', 'annotations')
            self._imgpath = osp.join(self.root, 'VisDrone2018-VID-train', 'sequences')
        else:
            self._annopath = osp.join(self.root, 'VisDrone2018-VID-val', 'annotations')
            self._imgpath = osp.join(self.root, 'VisDrone2018-VID-val', 'sequences')
            

        self.ids = list()

        rootpath = osp.join(self.root, 'VisDrone2018-VID-train', 'sequences')
        cnt = 0
        for foldername in os.listdir(rootpath):
            for item in os.listdir(osp.join(rootpath, foldername)):
                self.ids.append((os.path.join(rootpath, foldername), item.split('.')[0])) 
        print(1+len(self.ids))

        # for (year, name) in image_sets:
        #     rootpath = osp.join(self.root, 'VOC' + year)
        #     for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #         self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        # print('index: '+str(index))
        # print('img_info: '+str(self.ids[index]))

        im, gt, h, w = self.pull_item(index)
        while gt.shape == (1,5) and np.sum(gt, axis=1)==0:   # sum 哪一维，结果就不存在这一维
                                                             # shape 是小括号，取数是中括号
            index = (index + 1)%len(self.ids)
            im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_info = self.ids[index]
        # print('index: '+str(index))
        # print('img_info: '+str(img_info))
        img = cv2.imread(osp.join(self._imgpath, img_info[0], img_info[1]+'.jpg'), cv2.IMREAD_COLOR)
        height, width, channels = img.shape

        # target 检测目标框和类别，每一行代表一个obj，每一个target表示一张图中所有的框
        target = self.target_transform(img_info[0].replace('sequences', 'annotations'), int(img_info[1]), width, height)    # 读取坐标时转为小数，相对坐标
        # print('size of target' + str(np.array(target).shape))
        if self.transform is not None:  # 数据增强
            target = np.array(target)
            if target.shape[0] == 0: 
                # 处理一个frame没有bbox的情况
                print('bad data')
                return torch.from_numpy(img).permute(2,0,1), np.zeros([1,5]), height, width
            
            # print('size of target '+str(target.shape))
            # print('size of target1 '+str(target[:,:4].shape))
            # print('size of target2 '+str(target[:,4].shape))
            # with open('before.jpg', 'wb') as f:
            #     cv2.imwrite('before.jpg', img)
            # cv2.imshow('before', img)
            # cv2.imwrite('before.jpg', img)
            # print('before '+str(np.array(img).shape))

            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])   # 前四个+第五个
            # self.transform -> SSDAugmentation.call -> SSDAugmentation.augment -> compose.call

            # # 显示图片的代码
            # origin=np.copy(img)
            # imgheight, imgwidth, imgchannel=img.shape
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # origin=origin/origin.max()  # 对于浮点数必须在0-1.0之间才可以正常显示
            # for idx, it in enumerate(boxes):
            #     if float(it[0].item()) <= 1.0:
            #         origin=cv2.rectangle(origin, (int(it[0].item()*imgwidth), int(it[1].item()*imgheight)), (int(it[2].item()*imgwidth), int(it[3].item()*imgheight)), (255,255,0), 4)
            #         origin=cv2.putText(origin, DRONE_CLASSES[int(labels[idx])], (int(it[0].item()*imgwidth), int(it[1].item()*imgheight)-2), font, 1, (0,0,255), 1)
            #     else:
            #         origin=cv2.rectangle(origin, (int(it[0].item()), int(it[1].item())), (int(it[2].item()), int(it[3].item())), (255,255,0), 4)
            #         origin=cv2.putText(origin, DRONE_CLASSES[int(labels[idx])], (int(it[0].item()), int(it[1].item())-2), font, 1, (0,0,255), 1)
            # cv2.imshow('cat', origin)
            # cv2.waitKey()   # 加上waitkey才显示图片   

            # to rgb
            img = img[:, :, (2, 1, 0)]

            # print('img size: '+str(np.array(img).shape))
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1))) # box和labels水平堆叠, np.expand_dims可以使(numbox,) 转为(numbox, 1)
            # print('labels size: '+str(labels.shape))
            # print('boxes size: '+str(boxes.shape))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_info = self.ids[index]
        return cv2.imread(osp.join(self._imgpath, img_info[0], img_info[1]+'.jpg'), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_info = self.ids[index]
        gt = self.target_transform(img_info[0].replace('sequences', 'annotations'), int(img_info[1]))
        gt[[0, 4], :]=gt[[4, 0], :]
        return img_info, gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
