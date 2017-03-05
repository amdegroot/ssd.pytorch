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
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        features1: (nn.Sequential) VGG layers for input
            size of either 300 or 500
        phase: (string) Can be "test" or "train"
        size: (int) the SSD version for the input size. Can be 300 or 500.
            Defaul: 300
        num_classes: (int) the number of classes to score. Default: 21.
    """

    def __init__(self, phase, sz=300, num_classes=21):
        super(SSD, self).__init__()
        self.phase = phase
        self.size = sz
        self.num_classes = num_classes
        param=num_classes*3
        self.features1 = build_base(cfg[str(sz)] ,3)

        # TODO: Build the rest of the sequentials in a for loop.
        v = [0.1, 0.1, 0.2, 0.2] # variances
        ar = [1,1,2,1/2,3,1/3] # aspect ratios
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
        self.p4_3 = PriorBox(num_classes, sz, sz, 30, -1, ar[1:4], v, False, True)

        self.lfc7 = nn.Conv2d(1024,24,kernel_size=3,padding=1)
        self.cfc7 = nn.Conv2d(1024,param*2,kernel_size=3,padding=1)
        self.pfc7 = PriorBox(num_classes, sz, sz, 60, 114, ar, v, False, True)

        self.l6_2 = nn.Conv2d(512,24,kernel_size=3,padding=1)
        self.c6_2 = nn.Conv2d(512,param*2,kernel_size=3,padding=1)
        self.p6_2 = PriorBox(num_classes, sz, sz, 114, 168, ar, v, False, True)

        self.l7_2 = nn.Conv2d(256,24,kernel_size=3,padding=1)
        self.c7_2 = nn.Conv2d(256,param*2,kernel_size=3,padding=1)
        self.p7_2 = PriorBox(num_classes, sz, sz, 168, 222, ar, v, False, True)

        self.l8_2 = nn.Conv2d(256,24,kernel_size=3,padding=1)
        self.c8_2 = nn.Conv2d(256,param*2,kernel_size=3,padding=1)
        self.p8_2 = PriorBox(num_classes, sz, sz, 222, 276, ar, v, False, True)

        self.lp6 = nn.Conv2d(256,24,kernel_size=3,padding=1)
        self.cp6 = nn.Conv2d(256,param*2,kernel_size=3,padding=1)
        self.pp6 = PriorBox(num_classes, sz, sz, 276, 330, ar, v, False, True)

        self.softmax = nn.Softmax()
        self.detect = Detect(21, 0, 200, 0.01, 0.45, 400)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        x = self.features1(x)
        y = self.L2norm(x)
        # branch1 = [torch.transpose(torch.transpose(self.l4_3(y),1,2),2,3).contiguous(),torch.transpose(torch.transpose(self.c4_3(y),1,2),2,3).contiguous()]
        branch1 = [self.l4_3(y).permute(0,2,3,1), self.c4_3(y).permute(0,2,3,1)]
        p1 = self.p4_3(x)
        branch1 = [o.view(o.contiguous().size(0),-1) for o in branch1]
        x = self.features2(x)
        # branch2 = [torch.transpose(torch.transpose(self.lfc7(x),1,2),2,3).contiguous(),torch.transpose(torch.transpose(self.cfc7(x),1,2),2,3).contiguous()]
        branch2 = [self.lfc7(x).permute(0,2,3,1), self.cfc7(x).permute(0,2,3,1)]
        p2 = self.pfc7(x)
        branch2 = [o.view(o.contiguous().size(0),-1) for o in branch2]
        x = self.features3(x)
        # branch3 = [torch.transpose(torch.transpose(self.l6_2(x),1,2),2,3).contiguous(),torch.transpose(torch.transpose(self.c6_2(x),1,2),2,3).contiguous()]
        branch3 = [self.l6_2(x).permute(0,2,3,1), self.c6_2(x).permute(0,2,3,1)]
        p3 = self.p6_2(x)
        branch3 = [o.view(o.contiguous().size(0),-1) for o in branch3]
        x = self.features4(x)
        # branch4 = [torch.transpose(torch.transpose(self.l7_2(x),1,2),2,3).contiguous(),torch.transpose(torch.transpose(self.c7_2(x),1,2),2,3).contiguous()]
        branch4 = [self.l7_2(x).permute(0,2,3,1), self.c7_2(x).permute(0,2,3,1)]
        p4 = self.p7_2(x)
        branch4 = [o.view(o.contiguous().size(0),-1) for o in branch4]
        x = self.features5(x)
        # branch5 = [torch.transpose(torch.transpose(self.l8_2(x),1,2),2,3).contiguous(),torch.transpose(torch.transpose(self.c8_2(x),1,2),2,3).contiguous()]
        branch5 = [self.l8_2(x).permute(0,2,3,1), self.c8_2(x).permute(0,2,3,1)]
        p5 = self.p8_2(x)
        branch5 = [o.view(o.contiguous().size(0),-1) for o in branch5]
        x = self.pool6(x)
        # branch6 = [torch.transpose(torch.transpose(self.lp6(x),1,2),2,3).contiguous(),torch.transpose(torch.transpose(self.cp6(x),1,2),2,3).contiguous()]
        branch6 = [self.lp6(x).permute(0,2,3,1), self.cp6(x).permute(0,2,3,1)]
        p6 = self.pp6(x)
        branch6 = [o.view(o.contiguous().size(0),-1) for o in branch6]
        loc_layers = torch.cat((branch1[0],branch2[0],branch3[0],branch4[0],branch5[0],branch6[0]),1)
        conf_layers = torch.cat((branch1[1],branch2[1],branch3[1],branch4[1],branch5[1],branch6[1]),1)
        box_layers = torch.cat((p1,p2,p3,p4,p5,p6), 2)

        if self.phase == "test":
            conf_layers = conf_layers.view(-1,21)
            conf_layers = self.softmax(conf_layers)
            output = self.detect(loc_layers,conf_layers,box_layers)
        else:
            conf_layers = conf_layers.view(conf_layers.size(0),-1,self.num_classes)
            output = [loc_layers, conf_layers, box_layers]
        return output


    # This function is very closely adapted from jcjohnson pytorch-vgg conversion script
    # https://github.com/jcjohnson/pytorch-vgg/blob/master/t7_to_state_dict.py
    def load_weights(self, base_file, norm_file = './weights/normWeights.t7'):
        py_modules = list(self.modules())
        next_py_idx = 0
        scale_weight = load_lua(norm_file).float()
        other, ext = os.path.splitext(base_file)
        if ext == '.t7':
            print('Loading lua model weights...')
            other = load_lua(base_file)
        else:
            print('Only .t7 is supported for now.')
            return
        #elif: ext == ''
        for i, t7_module in enumerate(other.modules):
            if not hasattr(t7_module, 'weight'):
                continue
            assert hasattr(t7_module, 'bias')
            while not hasattr(py_modules[next_py_idx], 'weight'):
                next_py_idx += 1
            py_module = py_modules[next_py_idx]
            next_py_idx += 1

            # The norm layer should be the only layer with 1d weight
            if(py_module.weight.data.dim() == 1):
                # print('%r Copying data from\n  %r to\n  %r' % (i-1, "L2norm", py_module))
                # py_module.weight.data.copy_(scale_weight)
                py_module = py_modules[next_py_idx]
                next_py_idx += 1
            assert(t7_module.weight.size() == py_module.weight.size())
            print('%r Copying data from\n  %r to\n  %r' % (i, t7_module, py_module))

            py_module.weight.data.copy_(t7_module.weight)
            assert(t7_module.bias.size() == py_module.bias.size())
            py_module.bias.data.copy_(t7_module.bias)
        py_modules[-14].weight.data.copy_(scale_weight)
        print('%r Copying data from\n  %r to\n  %r' % (i-1, "L2norm", py_modules[-14]))


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def build_base(cfg, i, batch_norm=False):
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
    "300" : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512],
}


def build_ssd(phase, size, num_classes):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Sorry only SSD300 is supported currently!")
    return SSD(phase, size, num_classes)
