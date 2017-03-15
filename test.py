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
from data import AnnotationTransform, VOCDetection, test_transform
from timeit import default_timer as timer
import argparse
import numpy as np
from ssd import build_ssd

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/', type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.6, type=float, help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int, help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
args = parser.parse_args()

args.trained_model = 'ssd_models/123.pth'
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
    

def get_labelname(labelmap, top_label_indices):
    return [labelmap[int(l)-1] for l in top_label_indices]


def test_net(save_folder, net, cuda, valset, transform, top_k, thresh):
    for i in range(len(valset)):

        img = valset.pull_image(i)
        annotation = valset.pull_anno(i)
        x = Variable(transform(img).unsqueeze_(0)) # wrap tensor in Variable
        #
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data.cpu().numpy()
        # Parse the outputs.
        det_label = detections[0,:,1]
        det_conf = detections[0,:,2]
        det_xmin = detections[0,:,3]
        det_ymin = detections[0,:,4]
        det_xmax = detections[0,:,5]
        det_ymax = detections[0,:,6]
        # Get detections with confidence higher than thresh param.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= thresh and i < top_k]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices]
        top_labels = get_labelname(labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        for i in range(top_conf.shape[0]):
            if i>top_k:
                break
            xmin = int(round(top_xmin[i] * img.size[0]))
            ymin = int(round(top_ymin[i] * img.size[1]))
            xmax = int(round(top_xmax[i] * img.size[0]))
            ymax = int(round(top_ymax[i] * img.size[1]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            coords = (xmin, ymin, xmax-xmin, ymax-ymin)
            filename = save_folder+label_name+'.txt'
            with open(filename, mode='a') as f:
                print('GROUND TRUTH: {}  ||| PREDICTION: label: {} score: {} coords: {} {} {} {}'.format(annotation, label_name, score, *coords))
            f.closed


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
        cuda=True
    # evaluation
    test_net(args.save_folder, net, cuda, valset, test_transform(net.size,(104,117,123)), args.top_k, thresh=args.confidence_threshold)
