"""Custom meters for monitoring Precision and Recall metrics

Ellis Brown, Max deGroot
"""

import torch
import numpy as np
import math
from . import mAPmeter, VOC07APMeter


class VOC07mAPMeter(mAPmeter.mAPMeter):
    """Mean average precision meter for PASCAL V0C 2007 dataset """

    def __init__(self, ovp_thresh=0.5, use_difficult=False, class_names=None, pred_idx=0):
        """
        Param:
            ovp_thresh (optional, float): overlap threshold for TP
                (default: 0.5)
            use_difficult (optional, boolean) use difficult ground-truths if
                applicable, otherwise just ignore
                (default: False)
            class_names (optional, list of str): if provided, will print out AP
                for each class
                (default: None)
            pred_idx (optional, int): prediction index in network output list
                (default: 0)
        """
        self.apmeter = VOC07APMeter(
            ovp_thresh, use_difficult, class_names, pred_idx)
