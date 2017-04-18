"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325

TODO: implement data_augment for training

Ellis Brown, Max deGroot
"""

import torch
from torchvision import transforms
import cv2
import numpy as np
# import torch_transforms


def random_sample():
    """Randomly sample the image by 1 of:
        - using entire original input image
        - sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
        - randomly sample a patch
    sample patch size is [0.1, 1] * size of original
    aspect ratio b/t .5 & 2
    keep overlap with gt box IF center in sampled patch

    TODO: ... all of this ...

    Return:
        transform (transform): the random sampling transformation
    """

    raise NotImplementedError


def train_transform():
    """Defines the squential transformations to the input image that help make
        the model more robust to various input object sizes and shapes during
        training.

    sample -> resize -> flip -> photometric (?)

    TODO: complete all steps of augmentation

    Return:
        transform (transform): the transformation to be applied to the the
        image
    """
    # SAMPLE - Random sample the image
    # sample = random_sample()

    # RESIZE to fixed size
    # resize = transforms.RandomSizedCrop(224)

    # apply photo-metric distortions https://arxiv.org/pdf/1312.5402.pdf
    # photmetric = None

    return transforms.Compose([
        # sample,
        # resize,
        transforms.RandomHorizontalFlip()
        # photmetric
    ])


class SwapChannel(object):
    """Transforms a tensorized image by swapping the channels as specified
    in the swap

    modifies the input tensor

    Arguments:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Arguments:
            image (Tensor): image tensor to be transformed
        Returns:
            a tensor with channels swapped according to swap
        """
        temp = image.clone()
        for i in range(3):
            temp[i] = image[self.swaps[i]]
        return temp


class BaseTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """
    def __init__(self, resize, rgb_means, swap = (2, 0, 1)):
        self.means = rgb_means
        self.resize = resize
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img):
        img = cv2.resize(np.array(img), (self.resize,
                                         self.resize)).astype(np.float32)
        img -= self.means
        img = img.transpose(self.swap)
        return torch.Tensor(img)
    # return transforms.Compose([
    #     transforms.Scale(dim),
    #     transforms.CenterCrop(dim),
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.mul(255)),
    #     transforms.Normalize(mean_values, (1, 1, 1)),
    #     SwapChannel(swap)
    # ])
