"""Contains the BatchIterator functionality

Ellis Brown
"""

import os

import torch
from torch.utils import data
from torch.autograd import Function
from torch.autograd import Variable

import torchvision
from torchvision import transforms

from voc import AnnotationTransform, VOCDetection
from config import VOCroot, BATCHES, SHUFFLE, WORKERS


class BatchIterator(object):
    """Controls batches on their way into the network

    Arguments:
        train_loader (DataLoader): loader for the training set
        val_loader (DataLoader): loader for the validation set
        test_loader (DataLoader): loader for the testing set
        batch_size (int): batch size
    """

    def __init__(self, train_loader, val_loader, test_loader, batch_size=BATCHES):
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
