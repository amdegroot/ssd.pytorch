from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
import collections
import itertools
from torch.autograd import Variable
from config import VOCroot
import torch.utils.data as data
from PIL import Image, ImageDraw
import sys
import os
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from data import AnnotationTransform, VOCDetection, detection_collate, test_transform
from modules import MultiBoxLoss
from ssd import build_ssd
from timeit import default_timer as timer


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--silent', default=True, type=bool, help='Turn off progress tracking per iteration')
parser.add_argument('--epoch_save', default=False, type=bool, help='Save model every epoch')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
# parser.add_argument('--final_model_path', default='models/trained_ssd.pth.tar',
#                     help='Location to save final model')
args = parser.parse_args()

# Model
net = build_ssd('train',300,21)
if args.cuda:
    net.cuda()
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
batch_size = args.batch_size

def train():
    net.train()
    train_loss = 0
    for epoch in range(args.epochs)
        # load train data
        dataset = VOCDetection(VOCroot, 'train', test_transform(ssd_dim, rgb_means), AnnotationTransform())
        # create batch iterator
        batch_iterator = iter(data.DataLoader(dataset,batch_size,shuffle=True,collate_fn=detection_collate))
        for batch_idx in range(len(dataset) / batch_size):
            images, targets = next(batch_iterator)

            if args.cuda():
                images = images.cuda()
            optimizer.zero_grad()

            images = Variable(images)
            out = net(images)
            loss = net.multibox.loss(out, targets)
            loss.backward()
            optimizer.step()

            print("Current loss: ", loss.data[0])
            #train_loss += loss.data[0]
            #print(train_loss/(batch_idx+1))

if __name__ == '__main__':
    train()
