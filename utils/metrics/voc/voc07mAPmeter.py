"""Custom meters for monitoring Precision and Recall metrics

Ellis Brown, Max deGroot
"""

import torch
from . import meter
import numpy as np
import math

class VOC07mAPMeter(mAPMeter):
    """ Mean average precision metric for PASCAL V0C 07 dataset """
    def __init__(self, *args, **kwargs):
        super(VOC07mAPMeter, self).__init__(*args, **kwargs)

    # def add(self,rec,prec):
    #     """
    #    calculate average precision, override the default one,
    #    special 11-point metric
    #    Params:
    #    ----------
    #    rec : numpy.array
    #        cumulated recall
    #    prec : numpy.array
    #        cumulated precision
    #    Returns:
    #    ----------
    #    ap as float
    #    """
    #
