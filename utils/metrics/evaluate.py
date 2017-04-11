from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot
from data import VOC_CLASSES as labelmap
import torch.utils.data as data
from PIL import Image
import sys
import os
from data import AnnotationTransform, VOCDetection, base_transform
from timeit import default_timer as timer
import argparse
import numpy as np
from ssd import build_ssd
import pickle
from collections import Counter, defaultdict

from utils.box_utils import jaccard, point_form

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/',
                    type=str, help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.6,
                    type=float, help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()

args.trained_model = 'weights/ssd_300_VOC0712.pth'
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def voc_ap(rec, prec):
    """VOC_AP average precision calculations using 11-recall-point based AP
    metric (VOC2007)
    [precision integrated to recall]
    Params:
        rec (FloatTensor): recall cumsum
        prec (FloatTensor): precision cumsum
    Return:
        average precision (float)
    """
    ap = 0.
    for threshold in torch.range(0., 1., 0.1):
        if torch.sum(rec >= threshold) == 0:  # if no recs are >= this thresh
            p = 0
        else:
            # largest prec where rec >= thresh
            p = torch.max(prec[rec >= threshold])
        ap += p / 11.
    return ap


def test_net(net, cuda, valset, transform, top_k):
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(valset)
    ovthresh = 0.5
    num_classes = 0

    # per class
    fp = defaultdict(list)
    tp = defaultdict(list)
    gts = defaultdict(list)
    precision = Counter()
    recall = Counter()
    ap = Counter()

    for i in range(num_images):
        confidence_threshold = 0.01
        print('Evaluating image {:d}/{:d}....'.format(i + 1, num_images))
        img = valset.pull_image(i)
        anno = valset.pull_anno(i)
        # print(anno)
        anno = torch.Tensor(anno).long()
        gt_classes = list(set(anno[:, 4]))
        x = Variable(transform(img).unsqueeze_(0))
        if cuda:
            x = x.cuda()
        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.size[0], img.size[1],
                              img.size[0], img.size[1]])
        # for each class
        if num_classes == 0:
            num_classes = detections.size(1)
        for cl in range(detections.size(1)):
            dets = detections[0, cl, :, :]
            mask = dets[:, 0].ge(0.01).expand(5, dets.size(0)).t()
            # all dets w > 0.01 conf for class
            dets = torch.masked_select(dets, mask).view(-1, 5)
            mask = anno[:, 4].eq(cl).expand(5, anno.size(0)).t()
            # all gts for class
            truths = torch.masked_select(anno, mask).view(-1, 5)
            if truths.numel() > 0:
                truths = truths[:, :-1]
                gts[cl].extend([1] * truths.size(0))  # count gts
                if dets.numel() < 1:
                    continue  # no detections to count
                # there exist gt of this class in the image
                # check for tp & fp
                preds = dets[:, 1:]
                preds *= scale.unsqueeze(0).expand_as(preds)
                # compute overlaps
                overlaps = jaccard(truths.float(), preds)
                # if each gt obj is found yet
                found = [False] * overlaps.size(0)
                maxes = overlaps.max(0)
                for pb in range(overlaps.size(1)):
                    max_overlap = maxes[0][0, pb]
                    gt = maxes[1][0, pb]
                    if max_overlap > ovthresh:  # 0.5
                        if found[gt]:
                            # duplicate
                            fp[cl].append(1)
                            tp[cl].append(0)
                        else:
                            # not yet found
                            tp[cl].append(1)
                            fp[cl].append(0)
                            found[gt] = True  # mark gt as found
                    else:
                        fp[cl].append(1)
                        tp[cl].append(0)
            else:
                # there are no gts of this class in the image
                # all dets > 0.01 are fp
                if dets.numel() > 0:
                    fp[cl].extend([1] * dets.size(0))
                    tp[cl].extend([0] * dets.size(0))
    for cl in range(num_classes):
        # for each class calc rec, prec, ap
        tp_cumsum = torch.cumsum(torch.Tensor(tp[cl]), 0)
        fp_cumsum = torch.cumsum(torch.Tensor(fp[cl]), 0)
        gt_cumsum = torch.cumsum(torch.Tensor(gts[cl]), 0)
        pos_det = max(tp_cumsum) + max(fp_cumsum)
        # precision (tp / tp+fp)
        # recall (tp+fp / #gt) => gt = tp + fn
        precision[cl] = max(tp_cumsum) * 1.0 / pos_det
        recall[cl] = pos_det * 1.0 / max(gt_cumsum)
        # avoid div by 0 with .clamp(min=1e-12)
        rec = tp_cumsum / gt_cumsum.clamp(min=1e-12)
        prec = tp_cumsum / (tp_cumsum + fp_cumsum).clamp(min=1e-12)
        ap[cl] = voc_ap(rec, prec)
        recall[cl] = max(rec)
        precision[cl] = max(prec)
        print('class', cl, 'rec', recall[cl], 'prec', precision[cl], 'AP', ap[cl])
    # mAP = mean of APs for all classes
    mAP = sum(ap.values()) / len(ap)
    return mAP

if __name__ == '__main__':
    # load net
    net = build_ssd('test', 300, 21)    # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    valset = VOCDetection(VOCroot, 'val', None, AnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, valset, base_transform(
        net.size, (104, 117, 123)), args.top_k, thresh=args.confidence_threshold)
