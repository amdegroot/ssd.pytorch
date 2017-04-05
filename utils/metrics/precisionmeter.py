import torch
from . import meter
import numpy as np
import math

class PrecisionMeter(meter.Meter):
    def __init__(self, threshold = [0.5], perclass=False):
         self.threshold = sorted(thresholds)
         assert(self.threshold[0]>=0 and self.threshold[-1]<=1, 'threshold should be between 0 and 1')
         self.perclass = perclass
         self.reset()

     def reset(self):
         self.tp = {} # true positives
         self.tpfn = {} # true positives & false negatives
         for t in self.threshold:
             self.tp[t] = torch.Tensor()
             self.tpfn[t] = torch.Tensor()

     def add(self,output,target):
         output = output.squeeze()
         if output.dim() == 1:
             # unsqueeze zero dim
            output = output.view(1,output.size(0))
        else:
            assert(output:dim() == 2,'wrong output size (1D or 2D expected)')
        if target.dim() == 1:
            target = target.view(1, target.size(0))
        else:
            assert(target.dim() == 2,'wrong target size (1D or 2D expected)')
        for i,s in enumerate(output.size()):
            assert(s == target.size(i),'target and output do not match on dim %d'%(i))


        # initialize upon receiving first inputs
        for t in self.threshold:
            if self.tp[t].numel() == 0:
                self.tp[t].resize_(target.size(1)).type_as(output).fill_(0)
                self.tpfn[t].resize_(target.size(1)).type_as(output).fill_(0)

        # scores of true positives
        true_pos = output.clone()
        true_pos[target == 0] = -1

        # sum all the things
        for t in self.threshold:
            self.tp[t].add_(torch.ge(true_pos, t).type_as(output).sum(0).squeeze())
            self.tpfn[t].add_(torch.ge(output,t).type_as(output).sum(0).squeeze())

     def value(self,t=None):
         if t:
             assert(self.tp[t] and self.tpfn[t],'%f is an incorrect threshold [not set]'%(t))
             if self.perclass:
                 recallPerClass = (self.tp[t] / self.tpfn[t]) * 100
                 recallPerClass[self.tpfn[t] == 0] = 100
                 return recallPerClass
             else:
                 if self.tpfn[t].sum() == 0: return 100
                 return (self.tp[t].sum() / self.tpfn[t].sum()) * 100

          else:
             value = {}
             for t in self.threshold:
                value[t] = self.value(t)
             return value
