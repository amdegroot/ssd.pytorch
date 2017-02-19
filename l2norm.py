import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F
from f_l2norm import L2norm as l2norm
from torch.autograd import Variable


class L2norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2norm,self).__init__()
        self.n_channels = n_channels
        self.scale = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.weight.data.fill_(scale)
        # self.bias = nn.Parameter(torch.Tensor(n_channels))
        #self.gradWeight = nn.Parameter(torch.Tensor(self.n_channels))

    def forward(self, input):
        return l2norm(self.n_channels,self.scale)(input, self.weight)


    #
    # def reset():
    #     self.weight.fill_(self.scale)
