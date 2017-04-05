"""Meter for monitoring precision

inspired by
https://github.com/torchnet/torchnet/blob/f1d43f6a31d56072c88fecf4d255fca2dccc7458/meter/precisionmeter.lua

Ellis Brown, Max deGroot
"""

import torch
from . import meter
import numpy as np
import math


class PrecisionMeter(meter.Meter):
    """Precision Meter
    tracks percentage of true positives in all results retrieved

    precision = # true positive / # returned (true positives + false positives)
    """

    def __init__(self, threshold=[0.5], perclass=False):
        """
        Args:
            thresholds (optional, double list): list of thresholds [0,1] @ which
                the precision is measured
                (default: [0.5])
            perclass (optional, bool): measure precision per class if true,
                average over all examples if false
                (default: False)
        """
        # verify all thresholds between 0 & 1
        self.threshold = sorted(threshold)
        assert self.threshold[0] >= 0 and self.threshold[-1] <= 1, \
            'threshold should be between 0 and 1'
        self.perclass = perclass
        self.reset()

    def reset(self):
        """Resets the meter with empty member variables"""
        self.tp = {}  # true positives
        self.tpfp = {}  # true positives & false positives... all positive results
        for t in self.threshold:
            self.tp[t] = torch.Tensor()
            self.tpfp[t] = torch.Tensor()

    def add(self, output, target):
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
        """
        output = output.squeeze()
        if output.dim() == 1:
            # unsqueeze zero dim
            output = output.view(1, output.size(0))
        else:
            assert output.dim() == 2, 'wrong output size (1D or 2D expected)'
        if target.dim() == 1:
            target = target.view(1, target.size(0))
        else:
            assert target.dim() == 2, 'wrong target size (1D or 2D expected)'
        for i, s in enumerate(output.size()):
            assert s == target.size(i), \
                'target and output do not match on dim %d' % (i)

        # initialize upon receiving first inputs
        for t in self.threshold:
            if self.tp[t].numel() == 0:
                self.tp[t].resize_(target.size(1)).type_as(output).fill_(0)
                self.tpfp[t].resize_(target.size(1)).type_as(output).fill_(0)

        # scores of true positives
        true_pos = output.clone()
        true_pos[target == 0] = -1

        # sum all the things
        for t in self.threshold:
            self.tp[t].add_(torch.ge(true_pos, t).type_as(
                output).sum(0).squeeze())
            self.tpfp[t].add_(torch.ge(output, t).type_as(
                output).sum(0).squeeze())

    def value(self, t=None):
        """Returns the model's precision @ a specific threshold or all
        thresholds
        Note:
        - if t is not specified, returns a list containing the precision of the
        model predictions measured at all thresholds specified at initialization
        - if perclass was set True at initialization, the precision at each
        threshold will be a tensor of precisions per class instead of an average
        precision of all classes at the threshold (double)

        Args:
            t (optional, double): the threshold [0,1] for which the precision
                should be returned. Note: t must be a member of self.threshold
                (default: None)
        Return:
            precision:
                (double): the precision @ specified threshold
                (double dict): precision @ each t specified at initialization
        """

        if t:  # the precision @ specified threhsold
            assert self.tp[t] and self.tpfp[t], \
                '%f is an incorrect threshold [not set]' % (t)
            if self.perclass:
                prec_per_class = (self.tp[t] / self.tpfp[t]) * 100
                prec_per_class[self.tpfp[t] == 0] = 100
                return prec_per_class
            else:
                if self.tpfp[t].sum() == 0:
                    return 100
                return (self.tp[t].sum() / self.tpfp[t].sum()) * 100
        else:  # precision @ each threshold specified at initialization
            value = {}
            for t in self.threshold:
                value[t] = self.value(t)
            return value
