import torch
from . import meter
import numpy as np
import math

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
