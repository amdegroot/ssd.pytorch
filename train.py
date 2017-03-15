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
from data import VOCroot, v2,v1
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
from ssd_by_layer import build_ssd
from timeit import default_timer as timer
import time


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--epochs', default=500, type=int, help='Number of training epochs')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
args = parser.parse_args()

cfg = (v1,v2)[args.version == 'v2']

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

ssd_dim = 300 # only support 300 now
rgb_means = (104,117,123) # only support voc now
num_classes = 21
batch_size = args.batch_size
net = build_ssd('train',300,21)
vgg_weights = torch.load('weights/'+ args.basenet)
net.vgg.load_state_dict(vgg_weights)
if args.cuda:
    net.cuda()
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes,0.5,True,0,True,3,0.5, False)
def train():
    net.train()
    train_loss = 0
    dataset = VOCDetection(VOCroot, 'train',test_transform(ssd_dim, rgb_means), AnnotationTransform())

    for epoch in range(args.epochs):
        # load train data & create batch iterator
        batch_iterator = iter(data.DataLoader(dataset,batch_size,shuffle=True,collate_fn=detection_collate))
        adjust_learning_rate(optimizer, epoch)

        for iteration in range(len(dataset) // batch_size):
            images, targets = next(batch_iterator)
            if args.cuda:
                images = images.cuda()
                targets = [anno.cuda() for anno in targets]

            images = Variable(images)
            targets = [Variable(t) for t in targets]
            #forward
            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l,loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            print('Timer: ',t1-t0)
            if args.log_iters:
                print(repr(iteration) + ": Current loss: ", loss.data[0])
            train_loss += loss.data[0]
        train_loss/= (len(dataset) / batch_size)
        torch.save(net.state_dict(),'ssd_models/'+repr(epoch)+'.pth')
        print('Avg loss for epoch '+repr(epoch)+': '+repr(train_loss))
    torch.save(net,args.save_folder+''+args.version+'.pth')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 70 epochs
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (0.1 ** (epoch // 70))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
