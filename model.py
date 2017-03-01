import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.autograd import Function
from utils import*
from detection_output import Detect
from prior_box import PriorBox
from l2norm import L2norm as L2norm
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.serialization import load_lua
from f_l2norm import L2norm as norm
import torch.backends.cudnn as cudnn


class SSD(nn.Module):
    def __init__(self, features1,num_classes):
        super(SSD, self).__init__()
        param=num_classes*3
        self.features1 = features1
        self.features2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(512,1024,kernel_size=3,padding=6,dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024,1024,kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.features3 = nn.Sequential(
            nn.Conv2d(1024,256,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True),
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(512,128,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True),
        )
        self.features5 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool6 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3,stride=1),
        )

        self.L2norm = L2norm(512,20)
        self.l4_3 = nn.Conv2d(512,12,kernel_size=3,padding=1)
        self.c4_3 = nn.Conv2d(512,param,kernel_size=3,padding=1)
        self.p4_3 = PriorBox(num_classes, 300, 300, 30, -1, [1,2,1/2], [0.1, 0.1, 0.2, 0.2], False, True)
        self.lfc7 = nn.Conv2d(1024,24,kernel_size=3,padding=1)
        self.cfc7 = nn.Conv2d(1024,param*2,kernel_size=3,padding=1)
        self.pfc7 = PriorBox(num_classes, 300, 300, 60, 114, [1,1,2,1/2,3,1/3], [0.1, 0.1, 0.2, 0.2], False, True)
        self.l6_2=nn.Conv2d(512,24,kernel_size=3,padding=1)
        self.c6_2=nn.Conv2d(512,param*2,kernel_size=3,padding=1)
        self.p6_2=PriorBox(num_classes, 300, 300, 114, 168, [1,1,2,1/2,3,1/3], [0.1, 0.1, 0.2, 0.2], False, True)
        self.l7_2=nn.Conv2d(256,24,kernel_size=3,padding=1)
        self.c7_2=nn.Conv2d(256,param*2,kernel_size=3,padding=1)
        self.p7_2=PriorBox(num_classes, 300, 300, 168, 222, [1,1,2,1/2,3,1/3], [0.1, 0.1, 0.2, 0.2], False, True)
        self.l8_2=nn.Conv2d(256,24,kernel_size=3,padding=1)
        self.c8_2=nn.Conv2d(256,param*2,kernel_size=3,padding=1)
        self.p8_2=PriorBox(num_classes, 300, 300, 222, 276, [1,1,2,1/2,3,1/3], [0.1, 0.1, 0.2, 0.2], False, True)
        self.lp6 = nn.Conv2d(256,24,kernel_size=3,padding=1)
        self.cp6 = nn.Conv2d(256,param*2,kernel_size=3,padding=1)
        self.pp6=PriorBox(num_classes, 300, 300, 276, 330, [1,1,2,1/2,3,1/3], [0.1, 0.1, 0.2, 0.2], False, True)
        self.softmax = nn.Softmax()
        self.detect = Detect(21, True, 0, 'CENTER', False, 200, 0.01, 0.45, 400, False, 1)

    def forward(self, x, phase):
        x = self.features1(x)
        y = self.L2norm(x)
        branch1 = [torch.transpose(torch.transpose(self.l4_3(y),1,2),2,3).contiguous(),torch.transpose(torch.transpose(self.c4_3(y),1,2),2,3).contiguous()]
        p1 = self.p4_3(x)
        branch1 = [o.view(o.size(0),-1) for o in branch1]
        x = self.features2(x)
        branch2 = [torch.transpose(torch.transpose(self.lfc7(x),1,2),2,3).contiguous(),torch.transpose(torch.transpose(self.cfc7(x),1,2),2,3).contiguous()]
        p2 = self.pfc7(x)
        branch2 = [o.view(o.size(0),-1) for o in branch2]
        x = self.features3(x)
        branch3 = [torch.transpose(torch.transpose(self.l6_2(x),1,2),2,3).contiguous(),torch.transpose(torch.transpose(self.c6_2(x),1,2),2,3).contiguous()]
        p3 = self.p6_2(x)
        branch3 = [o.view(o.size(0),-1) for o in branch3]
        x = self.features4(x)
        branch4 = [torch.transpose(torch.transpose(self.l7_2(x),1,2),2,3).contiguous(),torch.transpose(torch.transpose(self.c7_2(x),1,2),2,3).contiguous()]
        p4 = self.p7_2(x)
        branch4 = [o.view(o.size(0),-1) for o in branch4]
        x = self.features5(x)
        branch5 = [torch.transpose(torch.transpose(self.l8_2(x),1,2),2,3).contiguous(),torch.transpose(torch.transpose(self.c8_2(x),1,2),2,3).contiguous()]
        p5 = self.p8_2(x)
        branch5 = [o.view(o.size(0),-1) for o in branch5]
        x = self.pool6(x)
        branch6 = [torch.transpose(torch.transpose(self.lp6(x),1,2),2,3).contiguous(),torch.transpose(torch.transpose(self.cp6(x),1,2),2,3).contiguous()]
        p6 = self.pp6(x)
        branch6 = [o.view(o.size(0),-1) for o in branch6]
        loc_layers = torch.cat((branch1[0],branch2[0],branch3[0],branch4[0],branch5[0],branch6[0]),1)
        conf_layers = torch.cat((branch1[1],branch2[1],branch3[1],branch4[1],branch5[1],branch6[1]),1)
        box_layers = torch.cat((p1,p2,p3,p4,p5,p6), 2)

        if phase == "test":
            conf_layers = conf_layers.view(-1,21)
            conf_layers = self.softmax(conf_layers)
            output = self.detect(loc_layers,conf_layers,box_layers)
        else:
            conf_layers = conf_layers.view(conf_layers.size(0),-1,21)
        return output


def make_layers(cfg, i,  batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512],
}

def build_ssd():
    return SSD(make_layers(cfg['A'],3),21)
