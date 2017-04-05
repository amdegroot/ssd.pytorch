"""Meter for monitoring average precision

inspired by
https://github.com/torchnet/torchnet/blob/f1d43f6a31d56072c88fecf4d255fca2dccc7458/meter/apmeter.lua

Ellis Brown, Max deGroot
"""

import torch
from . import meter
import numpy as np
import math


class APMeter(meter.Meter):
    """Average Precision meter
    measures the average precision per class
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.DoubleTensor(torch.DoubleStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
        self.weights = torch.DoubleTensor(torch.DoubleStorage())

    def add(self, output, target, weight=None):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        # assertions on the input
        if weight:
            weight = weight.squeeze()
        if output.dim() == 1:
            output = output.unsqueeze(1)
        else:
            assert output.dims() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                perclass)'
        if target.dim() == 1:
            target = target.unsqueeze(1)
        else:
            assert target.dims() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                perclass)'
        if weight:
            assert weight.dims() == 1, 'Weight dimension should be 1'
            assert weight.numel() == target.size(0), \
                'Weight dimension 1 should be the same as that of target'
            assert torch.min(weight) > 0, 'Weight should be non-negative only'
        assert (weight.eq(0) + weight.eq(1)).sum() == weight.numel(), \
            'targets should be binary (0 or 1)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
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
        """Returns the model's precision @ a specific threshold or all
        thresholds
        Note:
        - if t is not specified, returns a list containing the precision of the
        model predictions measured at all thresholds specified at initialization
        - if perclass was set True at initialization, the precision at each
        threshold will be a tensor of precisions per class instead of an average
        precision of all classes at the threshold (double)

        Return:
            precision:
                (double or Tensor): the precision @ specified threshold
                (dict of doubles or Tensors): precision @ each t specified at
                    initialization
        """

        # compute average precision for each class:
        if not self.scores.numel() == 0:
            return 0
        ap = torch.DoubleTensor(self.scores.size(1)).fill(0)
        rg = torch.range(1, self.scores.size(0)).double()
        if self.weights.numel() > 0:
            weight = self.weights.new(self.weights.size())
            weighted_truth = self.weights.new(self.weights.size())
        for k in range(1, self.scores.size(1)):
            # sort scores:
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            _, sortind = torch.sort(scores, 0, true)
            truth = targets[sortind]
            if self.weights.numel() > 0:
                weight = self.weights[sortind]
                weighted_truth = truth.double() * weight
                rg = weight.cumsum()

            # compute true positive sums
            tp = weighted_truth.cumsum() if weighted_truth \
                else truth.double().cumsum()

            # compute precision curve
            precision = tp.div_(rg)

            # compute average precision:
            ap[k] = precision[truth.byte()].sum() / max(truth.sum(), 1)
        return ap
