# batch_iterator.py
"""
augments image data and pairs with labels for loading into the network
"""

import os

import torch
from torch.utils import data
from torch.autograd import Function
from torch.autograd import Variable

import torchvision
from torchvision import transforms



# next method

class BatchIterator(data.DataLoader):
    """
    class for loading batch into network
    """

    def __init__(self, dataset):
        """

        Args:
            dataset (string): the desired dataset to load
        """

    def load(self, batch_size):
        return batch_size


class Dataset(data.Dataset):
    """Class representing the dataset

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
