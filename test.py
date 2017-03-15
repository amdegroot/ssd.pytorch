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

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/', type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.6, type=float, help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int, help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
args = parser.parse_args()

args.trained_model = 'weights/ssd_300_VOC0712.pth'
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)



def get_labelname(labelmap, top_label_indices):
    return [labelmap[int(l)-1] for l in top_label_indices]


def test_net(save_folder, net, cuda, valset, transform, top_k, thresh):

    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test1.txt'
    num_images = len(valset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1,num_images))
        img = valset.pull_image(i)
        img_id, annotation = valset.pull_anno(i)
        x = Variable(transform(img).unsqueeze_(0))

        with open(filename, mode='a') as f:
            f.write('GROUND TRUTH FOR: '+img_id+'\n')
            for box in annotation:
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')
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

        for j in range(top_conf.shape[0]):
            if j>top_k:
                break
            xmin = top_xmin[j] * img.size[0]
            ymin = top_ymin[j] * img.size[1]
            xmax = top_xmax[j] * img.size[0]
            ymax = top_ymax[j] * img.size[1]
            score = top_conf[j]
            label_name = top_labels[j]
            # print(img_id)
            coords = [xmin, ymin, xmax-xmin, ymax-ymin,score, img_id]
            with open(filename, mode='a') as f:
                f.write('PREDICTION: '+ repr(j)+ '\n')
                f.write('label: '+label_name+' score: '+str(score) +' '+' || '.join(str(c) for c in coords)+'\n\n')


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
    test_net(args.save_folder, net, args.cuda, valset, base_transform(net.size,(104,117,123)), args.top_k, thresh=args.confidence_threshold)
