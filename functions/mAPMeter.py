"""Custom meters for monitoring Precision and Recall metrics

Ellis Brown, Max deGroot
"""

import torch
from . import meter
import numpy as np
import math

class APMeter(meter.Meter):
    """Average Precision meter"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.scores = torch.DoubleTensor(torch.DoubleStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
        self.weights = torch.DoubleTensor(torch.DoubleStorage())

    def add(self, output,target,weight):
        if weight:
            weight = weight.squeeze()
        if output.dim() == 1:
            output = output.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1)
        if weight:
            assert(weight.numel()==target.size(0), 'Weight dimension 1 should be the same as that of target')
        if self.scores.numel()>0:
            assert(target.size(1)==self.targets.size(1), 'dimensions for output should match previously added examples.')

        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            new_weight_size = math.ceil(self.weights.storage().size() * 1.5)
            self.scores.storage().resize_(new_size + output.numel())
            self.targets.storage().resize_(new_size + output.numel())
            if weight:
                self.weights.storage().resize_(new_weight_size + output.size(0))
        # store scores and targets:
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(1, offset + 1, output.size(0)).copy_(output)
        self.targets.narrow(1, offset + 1, target.size(0)).copy_(target)

        if weight:
            self.weights.resize_(offset + weight.size(0))
            self.weights.narrow(1, offset + 1, weight.size(0)).copy_(weight)

    def value(self):
        # compute average precision for each class:
        if not self.scores.numel() == 0: return 0
        ap = torch.DoubleTensor(self.scores.size(1)).fill(0)
        range = torch.range(1, self.scores.size(0)).double()
        # weight, weightedtruth = 0
        if self.weights.numel() > 0:
            weight = self.weights.new(self.weights.size())
            weightedtruth = self.weights.new(self.weights.size())
        for k in range(self.scores.size(1))
            # sort scores:
            scores = self.scores[:,k]
            targets = self.targets[:,k]
            _,sortind = torch.sort(scores, 0, true)
            truth = targets[sortind]
            if self.weights.numel() > 0:
                weight = self.weights[sortind]
                weightedtruth = truth.double()*weight
                range = weight.cumsum()

            # compute true positive sums:
            tp =  weightedtruth.cumsum() if weightedtruth else truth.double().cumsum()

            # compute precision curve:
            precision = tp.div_(range)

            # compute average precision:
            ap[k] = precision[truth.byte()].sum() / max(truth.sum(), 1)
        return ap

class RecallMeter(meter.Meter):
    """Recall Meter
    tracks percentage of the class that the system correctly retrieves

    recall = num returned true / num gt true
    """

    def __init__(self, thresholds=[0.5], perclass=False):
        """
        Args:
            thresholds (optional, double list): list of thresholds [0,1] at which
                therecall is measured
                (default: [0.5])
            perclass (optional, bool): measure recall per class if true, average
                over all examples if false
                (default: False)
        """
        # verify all thresholds between 0 & 1
        for thresh in thresholds:
            assert(thresh >= 0 and thresh <= 1,
                'threshold should be between 0 and 1')
        self.thresholds = thresholds.sort()
        self.perclass = perclass
        self.reset()

    def reset(self):
        self.tp = {} # true positives
        self.tpfn = {} # true positives & false negatives
        for _,t in

    def add(self):
        pass

    def value(self):
        pass

class mAPMeter(meter.Meter):
    """Mean Average Precision Meter"""
    def __init__(self, ovp_thresh=0.5, use_difficult=False, class_names=None, pred_idx=0):
        self.apmeter = APMeter()

    def add(self, output, target, weight):
        self.apmeter.add(output,target,weight)

    def value(self):
        return self.apmeter.value().mean()

    def reset(self):
        self.apmeter.reset()
