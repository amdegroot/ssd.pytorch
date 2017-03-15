import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from box_utils import*
from data import v2, v1
from functions import Detect, PriorBox
from modules import L2Norm
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.serialization import load_lua
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
            size of either 300 or 512. Default: 300
        phase: (string) Can be "test" or "train"
        size: (int) the SSD version for the input size. Can be 300 or 500.
            Defaul: 300
        num_classes: (int) the number of classes to score. Default: 21.
    """

    def __init__(self, phase, version, sz=300, num_classes=21):
        super(SSD, self).__init__()
        self.phase = phase
        self.size = sz
        self.num_classes = num_classes
        param=num_classes*3
        self.base = build_base(cfg[str(sz)] ,3)
        self.version = v1 
        self.box_layer = PriorBox(self.version)
        self.priors = Variable(self.box_layer.forward())
        # TODO: Build the rest of the sequentials in a for loop.

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

        # Multibox layers (conv layers to learn features from different scales)
        self.L2Norm = L2Norm(512,20)

        self.l4_3 = nn.Conv2d(512,12,kernel_size=3,padding=1)
        self.c4_3 = nn.Conv2d(512,param,kernel_size=3,padding=1)

        self.lfc7 = nn.Conv2d(1024,24,kernel_size=3,padding=1)
        self.cfc7 = nn.Conv2d(1024,param*2,kernel_size=3,padding=1)

        self.l6_2 = nn.Conv2d(512,24,kernel_size=3,padding=1)
        self.c6_2 = nn.Conv2d(512,param*2,kernel_size=3,padding=1)

        self.l7_2 = nn.Conv2d(256,24,kernel_size=3,padding=1)
        self.c7_2 = nn.Conv2d(256,param*2,kernel_size=3,padding=1)

        self.l8_2 = nn.Conv2d(256,24,kernel_size=3,padding=1)
        self.c8_2 = nn.Conv2d(256,param*2,kernel_size=3,padding=1)

        self.lp6 = nn.Conv2d(256,24,kernel_size=3,padding=1)
        self.cp6 = nn.Conv2d(256,param*2,kernel_size=3,padding=1)

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
        x = self.base(x)
        y = self.L2Norm(x)

        b1 = [self.l4_3(y).permute(0,2,3,1), self.c4_3(y).permute(0,2,3,1)]
        b1 = [o.view(o.contiguous().size(0),-1) for o in b1]
        x = self.features2(x)

        b2 = [self.lfc7(x).permute(0,2,3,1), self.cfc7(x).permute(0,2,3,1)]
        b2 = [o.view(o.contiguous().size(0),-1) for o in b2]
        x = self.features3(x)

        b3 = [self.l6_2(x).permute(0,2,3,1), self.c6_2(x).permute(0,2,3,1)]
        b3 = [o.view(o.contiguous().size(0),-1) for o in b3]
        x = self.features4(x)

        b4 = [self.l7_2(x).permute(0,2,3,1), self.c7_2(x).permute(0,2,3,1)]
        b4 = [o.view(o.contiguous().size(0),-1) for o in b4]
        x = self.features5(x)

        b5 = [self.l8_2(x).permute(0,2,3,1), self.c8_2(x).permute(0,2,3,1)]
        b5 = [o.view(o.contiguous().size(0),-1) for o in b5]
        x = self.pool6(x)

        b6 = [self.lp6(x).permute(0,2,3,1), self.cp6(x).permute(0,2,3,1)]
        b6 = [o.view(o.contiguous().size(0),-1) for o in b6]
        loc = torch.cat((b1[0],b2[0],b3[0],b4[0],b5[0],b6[0]),1)
        conf = torch.cat((b1[1],b2[1],b3[1],b4[1],b5[1],b6[1]),1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0),-1,4),                     # loc preds
                self.softmax(conf.view(-1,self.num_classes)),   # conf preds
                self.priors                                     # default boxes
                )
        else:
            print(self.priors.size())
            conf = conf.view(conf.size(0),-1,self.num_classes)
            loc = loc.view(loc.size(0),-1,4)
            output = (loc, conf, self.priors)
        return output


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Error: Sorry Only .pth and .pkl files currently supported!')



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
    version = 'pool6'
    return SSD(phase, version, size, num_classes)
