import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco, visdrone, VOC_CLASSES, DRONE_CLASSES, COCO_CLASSES
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
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # self.cfg = (coco, voc)[num_classes == 21]   # 类似c++ ？语句，前面为中括号为false后面为true时的值
        if num_classes == len(DRONE_CLASSES)+1: # ignore class
            self.cfg = visdrone
        elif num_classes == len(VOC_CLASSES)+1:
            self.cfg = voc
        elif num_classes == len(COCO_CLASSES)+1:
            self.cfg =coco

        # TODO: check cfg
        print('SSD config----------')
        for k, v in self.cfg.items():   # .keys(), .values(), .items()
            print(' '+str(k)+': '+str(v))
            
        self.priorbox = PriorBox(self.cfg)
        # self.priors = Variable(self.priorbox.forward(), volatile=True)
        # # 建议volatile=True修改为
        with torch.no_grad():
            self.priors = torch.Tensor(self.priorbox.forward())
        

        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300]. or [batch, 3, res, res]

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
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)
        # 增加中间的l2norm， multi-scale， skip connection
        s = self.L2Norm(x)
        sources.append(s)   # 记录中间结果

        # 沿vgg计算
        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)   # 隔层做cache， source就是一个cache

        # 定位和分类在每一个cache上做，相当于skip connection，直接连接浅层特征图 
        # apply multibox head to source layers
        # TODO: 查看众多的loc和conf
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),   # batch, num of obj, 4维框
                conf.view(conf.size(0), -1, self.num_classes),  # batch, num of obj, 类别数量
                self.priors
            )

        # print('output 0 shape: '+str(output[0].shape))
        # print('output 1 shape: '+str(output[1].shape))
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
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
            in_channels = v # 输入通道数（上一层输出通道）
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    # 打印vgg每一层
    print('vgg layers----------')
    for idx, item in enumerate(layers):
        print('['+str(idx)+'] '+str(item))

    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    # print('in_channels:'+str(in_channels))
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v

        # print('in_channels'+str(in_channels))

    print('extra layers----------')
    for idx, item in enumerate(layers):
        print('['+str(idx)+'] '+str(item))
    print('\n')

    return layers

# 调用vgg和add_extra函数，返回vgg和增加的层作为参数输入multi-box
# multibox作用是增加卷积定位和分类的head层(loc, conf)，并返回所有层
# 即localization head和classification head， 分类和定位任务的最后一层，卷积特征图，产生结果
def multibox(vgg, extra_layers, cfg, num_classes):  # multibox的含义为多个检测框（boundingbox）
    """
    通过将每个边界框检测器分配到图像中的特定位置，one-stage目标检测算法（例如YOLO，SSD和DetectNet）都是这样来解决这个问题。
    因为，检测器学会专注于某些位置的物体。为了获得更好的效果，我们还可以让检测器专注于物体的形状和大小。
    """
    # print('vgg----------- ')
    # for idx, v in enumerate(vgg):
    #     print(str(idx)+': '+str(v))
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]   # loc和conf layer接在第21和倒2之后
    for k, v in enumerate(vgg_source):  # 使用mbox的前两个
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):   # 编号从start开始，但是遍历顺序不变。 [1::2] 从idx为1开始（第二个），隔一个取一个
        # loc和conf接在extra layer第二个和之后，隔一个接一个
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] # 使用mbox的之后几个
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]

    # 构建VGG，VGG多余的层，分类和回归层

    print('loc layers----------')
    for idx, item in enumerate(loc_layers):
        print('  ['+str(idx)+'] '+str(item))
    print('conf layers----------')
    for idx, item in enumerate(conf_layers):
        print('  ['+str(idx)+'] '+str(item))
    print('\n')
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256, 128, 256],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 6, 4, 4],
}


def build_ssd(phase, size=300, num_classes=len(VOC_CLASSES)):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    # if size != 300:
    #     print("ERROR: You specified size " + repr(size) + ". However, " +
    #           "currently only SSD300 (size=300) is supported!")
    #     return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
