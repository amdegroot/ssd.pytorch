"""Meter for monitoring recall

inspired by
https://github.com/torchnet/torchnet/blob/f1d43f6a31d56072c88fecf4d255fca2dccc7458/meter/recallmeter.lua

Ellis Brown, Max deGroot
"""

import torch
from . import meter
import numpy as np
import math


class RecallMeter(meter.Meter):
    """Recall Meter
    tracks percentage of the class that the system correctly retrieves

    recall = num returned true / num gt true
    """

    def __init__(self, thresholds=[0.5], perclass=False):
        """
        Args:
            thresholds (optional, double list): list of thresholds [0,1] @ which
                the recall is measured
                (default: [0.5])
            perclass (optional, bool): measure recall per class if true, average
                over all examples if false
                (default: False)
        """
        # verify all thresholds between 0 & 1
        self.threshold = sorted(thresholds)
        assert self.threshold[0] >= 0 and self.threshold[-1] <= 1, \
            'threshold should be between 0 and 1'
        self.perclass = perclass
        self.reset()

    def reset(self):
        """Resets the meter with empty member variables"""
        self.tp = {}  # true positives
        self.tpfn = {}  # true positives & false negatives
        for t in self.threshold:
            self.tp[t] = torch.Tensor()
            self.tpfn[t] = torch.Tensor()

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

        # initialize upon receiving first inputs:
        for t in self.threshold:
            if self.tp[t].numel() == 0:
                self.tp[t].resize_(target.size(1)).type_as(output).fill_(0)
                self.tpfn[t].resize_(target.size(1)).type_as(output).fill_(0)

        # scores of true positives:
        true_pos = output.clone()
        true_pos[target == 0] = -1

        # sum all the things:
        for t in self.threshold:
            self.tp[t].add_(torch.ge(true_pos, t).type_as(
                output).sum(0).squeeze())
            self.tpfn[t].add_(target.type_as(output).sum(0).squeeze())

    def value(self, t=None):
        """Returns the model's recall @ a specific threshold or all thresholds
        Note:
        - if t is not specified, returns a list containing the recall of the
        model predictions measured at all thresholds specified at initialization
        - if perclass was set True at initialization, the recall at each
        threshold will be a list of thresholds per class instead of an average
        Args:
            t (optional, double): the threshold [0,1] for which the recall
                should be returned. Note: t must be a member of self.threshold
                (default: None)
        Return:
            recall:
                (double): the recall @ specified threshold
                (double list): recall @ each t specified at initialization
        """

        if t:  # the recall @ specified threhsold
            assert self.tp[t] and self.tpfn[t], \
                '%f is an incorrect threshold [not set]' % (t)
            if self.perclass:
                recall_per_class = (self.tp[t] / self.tpfn[t]) * 100
                recall_per_class[self.tpfn[t] == 0] = 100
                return recall_per_class
            else:
                if self.tpfn[t].sum() == 0:
                    return 100
                return (self.tp[t].sum() / self.tpfn[t].sum()) * 100
        else:  # recall @ each threshold specified at initialization
            value = {}
            for t in self.threshold:
                value[t] = self.value(t)
            return value
