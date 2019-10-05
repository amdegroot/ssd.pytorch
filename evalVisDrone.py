"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
# from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, VOC_CLASSES, BaseTransform
from data import DRONE_ROOT, DroneAnnotationTransform, DroneDetection, BaseTransform
from data import DRONE_CLASSES as labelmap
import torch.utils.data as data

from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd512_VisDrone2018_12000.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--dataset', default='VisDrone2018', type=str,
                    help='VOC, VisDrone2018, COCO')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# annopath = os.path.join(DRONE_ROOT, 'VOC2007', 'Annotations', '%s.xml') # 参数路径
# imgsetpath = os.path.join(DRONE_ROOT, 'VOC2007', 'ImageSets',
#                           'Main', '{:s}.txt')


validpath=os.path.join(DRONE_ROOT, 'VisDrone2018-VID-val')                        
dataset_mean = (119, 122, 116)
cachedir = os.path.join(DRONE_ROOT, 'annotations_cache')
set_type = 'test'


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):

    clip=filename.split('@')[0]
    frameidx=int(filename.split('@')[1])

    objects = []

    with open(os.path.join(validpath, 'annotations', clip+'.txt'), 'r+') as f:
        lines=f.readlines()

    for line in lines:
        l=line.split(',')
        if int(l[0]) == frameidx:
            obj_struct={}
            obj_struct['name']=int(l[7])
            obj_struct['bbox']=[int(l[2]), int(l[3]), int(l[2])+int(l[4]), int(l[3])+int(l[5])]
            objects.append(obj_struct)
    
    # print('object num: '+str(len(objects)))
    return objects  # 第一维和图片数量相同，，第二维是一个dict，每个key对应的value是int

    # """ Parse a PASCAL VOC xml file """
    # tree = ET.parse(filename)
    # objects = []
    # for obj in tree.findall('object'):
    #     obj_struct = {}
    #     obj_struct['name'] = obj.find('name').text
    #     obj_struct['pose'] = obj.find('pose').text
    #     obj_struct['truncated'] = int(obj.find('truncated').text)
    #     obj_struct['difficult'] = int(obj.find('difficult').text)
    #     bbox = obj.find('bndbox')
    #     obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
    #                           int(bbox.find('ymin').text) - 1,
    #                           int(bbox.find('xmax').text) - 1,
    #                           int(bbox.find('ymax').text) - 1]
    #     objects.append(obj_struct)
    # return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir

# 获得保存的结果的文件名和路径
def get_results_file_path(phrase, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + phrase + '_%s.txt' % (cls)
    filedir = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_results_file(all_boxes, dataset, imagenames): # all_boxes第一维对应类别，第二维对应哪个图片，第三维是检测框的数量，最后一维是5(x0,y0,x1,y1,conf)


    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} results file'.format(cls))
        filename = get_results_file_path(set_type, cls) # 获取保存的结果的文件名
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(imagenames):    # 图片的序号和图片名
                dets = all_boxes[cls_ind+1][im_ind] # Nx5 N为检测框数量
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))
                    # 按类别表示每一个文件
                    # 图片编号，conf，x0，y0，x1，y1


def do_python_eval(imagenames, output_dir='output'):
    
    aps = []
    # The PASCAL VOC metric changed in 2010
    # use_07_metric = use_07
    # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # 遍历所有类别，计算每一类的ap
    for i, cls in enumerate(labelmap):
        if cls == 'others':
            continue
        filename = get_results_file_path(set_type, cls) # 预测结果保存的文件名，category-wise
        rec, prec, ap = drone_eval(
           filename, i+1, imagenames,  # set_type='test' 测试阶段。 从第一类class=1开始，0为ignored class
           ovthresh=0.5)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


# 根据准确率和召回率计算map
def drone_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# 对一个类别做eval，计算召回率，准确率和ap
def drone_eval(detpath,
             # annopath,  # GT annotation path
             # imagesetfile,  # 测试图片名的集合
             classname, # 测试的所有图片名的集合
             # cachedir,  # anno cache
             imagenames,
             ovthresh=0.5,
             # use_07_metric=True
             ):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    
    # 加载gt anno cache
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')

    # read list of images 读取所有图片名称
    # with open(imagesetfile, 'r') as f:
    #     lines = f.readlines()
    # imagenames = [x.strip() for x in lines] 



    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(imagename) # 读入xml文件每一个标注项，成为一个一张图片所有object和其标注的表
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save 把所有标注全部读入
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # print(recs) # dict key=imgname val=[{key-val},{key-val}]包括类别，框等信息
    # print('resc'+str(len(recs)))    # 2846

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    # imagename=foldername@imgname
    for imagename in imagenames:    # 测试数据集中每一张图片读取标签
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        # difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + len(R)
        class_recs[imagename] = {'bbox': bbox,
                                 'det': det}
    # print(len(R))

    # read dets 读取预测结果
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        # sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = drone_ap(rec, prec)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap

def get_image_names_list():
    id2path=[]
    for clip in sorted(os.listdir(os.path.join(validpath, 'sequences'))):
        for frame in sorted(os.listdir(os.path.join(validpath, 'sequences', clip))):
            imginfo=clip + '@' + frame.split('.')[0]    # split with @
            id2path.append(imginfo)

    print('id2path: '+str(len(id2path)))
    return id2path


def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    # output_dir = get_output_dir('ssd300_120000', set_type)
    # det_file = os.path.join(output_dir, 'detections.pkl')

    # num_images=1

    runtime=0.0
    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        if gt.shape == (1,5) and np.sum(gt, axis=1)==0:
            print('bad test data')
            continue

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        
        detections = net(x).data
        # print(detections.shape)   # [1, 21, 200, 5] 1output, 21class, 200？, 5(x1,y1,x2,y2,conf)
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]  # 一类的所有检测结果 [200, 5]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()    # 第一列
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets  # 所有扩大到正常大小之后的检测框
        if (i+1)%100==0:
            print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))
        runtime+=detect_time

    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, dataset)

    print('[Speed]: '+str(round(runtime/num_images, 4))+' s, '+str(round(num_images/runtime,1))+' fps')


def evaluate_detections(box_list, dataset):
    # image index to image path/global name
    imagenames=get_image_names_list()
    write_results_file(box_list, dataset, imagenames)   # 写预测结果到文件

    do_python_eval(imagenames)    # 执行eval

# TODO needs modification
if __name__ == '__main__':
    # load data
    # if args.dataset == 'VOC':
    #     args.data_root = VOC_ROOT
    #     labelmap = VOC_CLASSES
    #     dataset_mean = (104, 117, 123)
    #     dataset = VOCDetection(args.data_root, [('2007', set_type)],
    #                         BaseTransform(300, dataset_mean),
    #                         VOCAnnotationTransform())
    # elif args.dataset == 'VisDrone2018':
    # args.data_root = DRONE_ROOT
    dataset_mean = (119, 122, 116)
    dataset = DroneDetection(DRONE_ROOT, 
                        BaseTransform(512, dataset_mean), 
                        target_transform=DroneAnnotationTransform(), train=0)

    # load net
    num_classes = len(labelmap) + 1                      # +1 for background
    net = build_ssd('test', 512, num_classes)            # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data


    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, 512,
             thresh=args.confidence_threshold)
