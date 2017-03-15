"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325

TODO: explore https://github.com/ncullen93/torchsample/blob/master/torchsample/transforms
    for any useful tranformations
TODO: implement data_augment for training

Ellis Brown
"""

from torchvision import transforms
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


class SwapChannelTransform(object):
    """Transforms a tensorized image by swapping the channels as specified in the swap

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


def test_transform(dim, mean_values):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        dim (int): input dimension to SSD
        mean_values ( (int,int,int) ): average RGB of the dataset
            (104,117,123)

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    swap = (2, 1, 0)

    return transforms.Compose([
        transforms.Scale(dim),
        transforms.CenterCrop(dim),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        SwapChannelTransform(swap),
        transforms.Normalize(mean_values, (1, 1, 1))
    ])
