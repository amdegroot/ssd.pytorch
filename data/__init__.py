from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .config import *
import cv2
import numpy as np
import torch

def base_transform(image, size, means):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    min_val = x.min()
    x -= min_val
    max_val = x.max()
    x /= max_val
    mean = means[::-1].copy()
    mean -= min_val
    mean /= max_val
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, means):
        self.size = size
        self.means = np.array(means, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.means), boxes, labels