# data_loader.py
"""
contains functionality for fetching datasets for use in the network
"""

import os
import urllib


# https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
# https://github.com/mprat/pascal-voc-python/blob/master/voc_utils/voc_utils.py


class VOCFetcher(object):
    """Allows you to download the VOC dataset to your local

    Args:
        data_dir (string): path to directory where data should be stored
            default: '~/data'
    """

    # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    # http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    def __init__(self,data_dir=None):
        self.data_dir = data_dir or \
            os.path.join(os.path.expanduser("~"),'/data')

    def get2007(self,test=True):
        """Downloads the 2007 VOC trainval dataset (and testset if desired)
        into the data directory specified in the class constructor
        Args:
            true (boolean): whether or not to download the test set
        """
        base = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/'
        trainval = base + 'VOCtrainval_06-Nov-2007.tar'
        
        urllib.urlretrieve(trainval, self.data_dir)
