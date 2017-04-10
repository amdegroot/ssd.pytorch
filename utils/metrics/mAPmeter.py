import math
from . import meter, APMeter
import numpy as np
import torch


class mAPMeter(meter.Meter):
    """
    The mAPMeter measures the mean average precision over all classes.

    The mAPMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """
    def __init__(self):
        self.apmeter = APMeter()

    def reset(self):
        self.apmeter.reset()

    def add(self, output, target, weight=None):
        self.apmeter.add(output, target, weight)

    def value(self):
        return self.apmeter.value().mean()
