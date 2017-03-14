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
parser.add_argument('--image_file', default='data/example.jpg', type=str, help='image file path to open')
parser.add_argument('--trained_model', default='weights/', type=str, help='image file path to open')
args = parser.parse_args()
#
# img = cv2.imread(args.image_file)

image = Image.open(args.image_file).convert('RGB')

x = test_trasform()
x = Variable(x) # wrap tensor in Variable
y = net(x)      # forward pass



detections = y.data.cpu().numpy()
# Parse the outputs.
det_label = detections[0,:,1]
det_conf = detections[0,:,2]
det_xmin = detections[0,:,3]
det_ymin = detections[0,:,4]
det_xmax = detections[0,:,5]
det_ymax = detections[0,:,6]

label = labelmap[int(det_label[0])-1]
score = det_conf[0]
x1 = det_xmin[0]*image.size[0]
y1 = det_ymin[0]*image.size[1]
x2 = det_xmax[0]*image.size[0]
y2 = det_ymax[0]*image.size[1]

coords = (x1, y1), x2-x1+1, y2-y1+1
return label,coords

if __name__ == '__main__':
    # load data
    valset = VOCDetection(VOCroot, 'val', test_transform(ssd_dim, rgb_means), AnnotationTransform())
    for i in len(valset):
        valset.pull_image()
    # load net
    net = build_ssd('test', 300, 21)    # initialize SSD
    net.load_weights(args.trained_model)
    print('load model successfully!')

    net.cuda()
    net.eval()

    # evaluation
    test_net(save_name, net, valset, max_per_image, thresh=thresh, vis=vis)
