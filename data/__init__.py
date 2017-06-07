from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .config import *
import cv2
import numpy as np
import torch

def base_transform(image, means):
    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= x.min()
    x /= x.max()
    x -= (means[2], means[1], means[0])
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)
    return x