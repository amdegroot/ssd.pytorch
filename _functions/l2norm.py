import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function


class L2Norm(Function):
    def __init__(self, n_channels, scale):
        self.n_channels = n_channels
        self.scale = scale
    def forward(self, input, weight):
        self.save_for_backward(input, weight)
        output = input.new().resize_as_(input)
        norms = torch.norm(input[0],2,0)

        output.copy_(input)
        for n in range(0,input.size(0)):
            for c in range(0, input.size(1)):
                output[n][c].div_(norms)
                output[n][c].mul_(weight[c])
        return output


    def backward(self, grad_output):
        input, weight = self.saved_tensors
        grad_input = grad_weight = None

        if self.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if self.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        return grad_input, grad_weight
