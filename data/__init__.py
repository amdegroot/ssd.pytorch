from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .config import *
import cv2
import numpy as np
import torch

def base_transform(image, size, means):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= x.min()
    x /= x.max()
    x -= (means[2], means[1], means[0])
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, means):
        self.size = size
        self.means = means

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.means), boxes, labels