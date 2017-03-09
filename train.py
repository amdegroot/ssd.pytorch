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
from data import VOCroot
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
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
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
ssd_dim = 300
rgb_means = (104,117,123)
net = build_ssd('train',300,21)
if args.cuda:
    net.cuda()
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
batch_size = args.batch_size
criterion = MultiBoxLoss(21,0.5,True,0,True,3,0.5, False)
def train():
    net.train()
    train_loss = 0
    dataset = VOCDetection(VOCroot, 'train', test_transform(ssd_dim, rgb_means), AnnotationTransform())
    batch_iterator = iter(data.DataLoader(dataset,batch_size,shuffle=True,collate_fn=detection_collate))
    images, targets = next(batch_iterator)
    if args.cuda:
        images = images.cuda()
    images = Variable(images)
    targets = [Variable(anno).cuda() for anno in targets]
    torch.save(images, "train_batch1.pkl")
    torch.save(targets, "train_batch1_annos.pkl")
    for epoch in range(args.epochs):
        # load train data
        # create batch iterator
        for batch_idx in range(len(dataset) // batch_size):
            optimizer.zero_grad()
            out = net(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            print("Current loss: ", loss.data[0])
            train_loss += loss.data[0]
            #print(train_loss/(batch_idx+1))
        train_loss/= (len(dataset) / batch_size)
        print("Loss for epoch" '%d': ' %d', epoch, train_loss)
    torch.save(net,"memo_net1.pkl")

if __name__ == '__main__':
    train()
