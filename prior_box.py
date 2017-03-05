import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
import numpy as np
import math

class PriorBox(Function):
    def __init__(self, num_classes, imWidth, imHeight, min_size, max_size, aspect_ratios, variance, flip, clip):
        #super(PriorBox, self).__init__()
        self.min_size = min_size
        self.imWidth = imWidth
        self.imHeight = imHeight

        if self.min_size < 0:
            print('<PriorBox> must provide positive min_size')
            return
        self.max_size = max_size or -1
        self.aspect_ratios = aspect_ratios # provided # of aspect ratios correlates to # of different priors
        #asp_ratios = {1} -- always atleast this 1 default

        self.flip = flip
        num_priors = len(aspect_ratios)
        if self.max_size < self.min_size and self.max_size > 0:
            print('<PriorBox> provided max_size must be greater than min_size')
            num_priors = num_priors + 1
        self.clip = clip
        self.num_priors = num_priors
        self.variance = variance or [0.1]

        if len(self.variance) != 1 and len(self.variance) != 4:
            print('<PriorBox> must provide exactly 0 or 4 variances in table format')

        for v in variance:
            if v <= 0:
                print('<PriorBox> variances must be greater than 0')

    def forward(self, input):

        dims = input.dim()
        iheight = input.size(dims-2)
        iwidth = input.size(dims-1)
        self.iheight = iheight
        self.iwidth = iwidth

        output = torch.Tensor(1,2,self.iheight*self.iwidth*self.num_priors*4)
        step_x = self.imWidth / self.iwidth    # ratio of image width to layer width

        step_y = self.imHeight / self.iheight  # ratio of image height to layer height

        top_data = output[0]
        mean_coords = top_data[0]
        dim = self.iheight * self.iwidth * self.num_priors * 4
        idx = -1
        for h in range(0,self.iheight):
            for w in range(0,self.iwidth):
                center_x = ((w+1)-0.5) * step_x
                center_y = ((h+1)-0.5) * step_y

                #first prior aspect_ratio=1, size = min_size
                box_width, box_height = self.min_size, self.min_size
                idx+=1
                mean_coords[idx] = (center_x - box_width / 2) / self.imWidth
                idx +=1
                # ymin
                mean_coords[idx] = (center_y - box_height / 2) / self.imHeight
                idx +=1
                # xmax
                mean_coords[idx] = (center_x + box_width / 2) / self.imWidth
                idx +=1
                # ymax
                mean_coords[idx] = (center_y + box_height / 2) / self.imHeight
                if self.max_size > 0:
                    box_width = math.sqrt(self.min_size * self.max_size)
                    box_height = math.sqrt(self.min_size * self.max_size)
                    idx = idx + 1
                    # xmin
                    mean_coords[idx] = (center_x - box_width / 2) / self.imWidth
                    idx = idx + 1
                    # ymin
                    mean_coords[idx] = (center_y - box_height / 2) / self.imHeight
                    idx = idx + 1
                    # xmax
                    mean_coords[idx] = (center_x + box_width / 2) / self.imWidth
                    idx = idx + 1
                    # ymax
                    mean_coords[idx] = (center_y + box_height / 2) / self.imHeight
                for i in range(0,len(self.aspect_ratios)):
                    ar = self.aspect_ratios[i]
                    if not (abs(ar-1) < 1e-6):
                        box_width = self.min_size * math.sqrt(ar)
                        box_height = self.min_size / math.sqrt(ar)
                        idx+=1
                        # xmin

                        mean_coords[idx] = (center_x - box_width / 2) / self.imWidth
                        idx +=1
                        # ymin
                        mean_coords[idx] = (center_y - box_height / 2) / self.imHeight
                        idx +=1
                        # xmax
                        mean_coords[idx] = (center_x + box_width / 2) / self.imWidth
                        idx +=1
                        # ymax
                        mean_coords[idx] = (center_y + box_height / 2) / self.imHeight
        if self.clip:
            # TODO look at torch.clamp()
            clipper = lambda t: min(max(t,0),1)
            mean_coords.apply_(clipper)
        variance = torch.Tensor(self.variance)
        dim = dim//4
        top_data[1]= variance.repeat(dim)

        return output


        def backward(self, grad_output):
            return None
