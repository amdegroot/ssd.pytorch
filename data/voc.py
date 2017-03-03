import os
import os.path
import sys
import urllib
import torch.utils.data as data
from PIL import Image, ImageDraw
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from config import VOCroot

voc_img = os.path.join(VOCroot, '')

def list_image_sets():
    """List all the image sets from Pascal VOC. Don't bother computing
    this on the fly, just remember it. It's faster.
    """
    return ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car',
        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']





class VOCLoader(data.Dataset):

    def __init__(self, root, image):
        self.root = root
        self.image = image


class VOC(data.Dataset):
    """Class representing the VOC dataset

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """
    def __init__(self, data):
        self.data = data


    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

