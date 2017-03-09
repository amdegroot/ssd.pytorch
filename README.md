# SSD: Single Shot MultiBox Object Detector, in PyTorch
A [PyTorch](http://pytorch.org/) implementation of [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325) from the 2016 paper by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang, and Alexander C. Berg.  The official and original Caffe code can be found [here](https://github.com/weiliu89/caffe/tree/ssd).  


### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#a-datasets'>Datasets</a>
- <a href='#b-training-ssd'>Train/Eval</a>
- <a href='#c-use-a-pre-trained-ssd-network-for-detection'>Demos</a>
- <a href='#todo'>Future Work</a>
- <a href='#references'>Reference</a>

<img align="right" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/ssd.png" height = 400/>




## Installation
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository.
- Then download the dataset by following the instructions below.
- Note: For training, we currently only support [VOC](http://host.robots.ox.ac.uk/pascal/VOC/), but are adding [COCO](http://mscoco.org/) and hopefully [ImageNet](http://www.image-net.org/) soon.

## A. Datasets
To make things easy, we provide a simple VOC dataset loader that enherits `torch.utils.data.Dataset` making it fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).

### VOC Dataset
##### Download VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```
##### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

 Ensure the following directory structure (as specified in [VOCdevkit](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/devkit_doc_07-Jun-2007.pdf)):

```
VOCdevkit/                                  % development kit
VOCdevkit/VOC2007/ImageSets                 % image sets
VOCdevkit/VOC2007/Annotations               % annotation files
VOCdevkit/VOC2007/JPEGImages                % images
VOCdevkit/VOC2007/SegmentationObject        % segmentations by object
VOCdevkit/VOC2007/SegmentationClass         % segmentations by class
```

## B. Training SSD
- First download the [VGG-16](https://arxiv.org/abs/1409.1556) base network weights which can be found [here](https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth), courtesy of [Justin Johnson](https://github.com/jcjohnson/pytorch-vgg).
- To train SSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.
```Shell
python train.py
```
- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * This repo supports [TensorBoard](https://github.com/torrvision/crayon).

## C. Use a pre-trained SSD network for detection

### Download a pre-trained network
- We are trying to provide PyTorch `state_dicts` (dict of weight tensors) of the latest SSD model definitions trained on different datasets.  
- Currently, we provide the following PyTorch models: 
    * SSD300 trained on VOC0712 
      - https://s3.amazonaws.com/amdegroot-models/ssd_300_voc07.tar.gz 
    * SSD300 (pool6/non-bn version) trained on VOC07
      - https://s3.amazonaws.com/amdegroot-models/ssd_300_voc07.tar.gz
- Our goal is to reproduce this table from the [original paper](http://arxiv.org/abs/1512.02325) 
<p align="center">
<img src="http://www.cs.unc.edu/~wliu/papers/ssd_results.png" alt="SSD results on multiple datasets" width="800px"></p>

## Try the demo
- Make sure you have [jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html) installed.
- Two alternatives for installing jupyter notebook:
    1. If you installed PyTorch with [conda](https://www.continuum.io/downloads) (recommended), then you should already have it.  (Just  navigate to the ssd.pytorch cloned repo and run): 
    `jupyter notebook` 

    2. If using [pip](https://pypi.python.org/pypi/pip):
```Shell
# make sure pip is upgraded
pip3 install --upgrade pip
# install jupyter notebook
pip install jupyter
# Run this inside ssd.pytorch
jupyter notebook
```
- Now navigate to `demo.ipynb` in the browser window that pops up and have at it!

## Testing
To evaluate a trained network:
```Shell
python test.py
```
You can specify the parameters listed in the `test.py` file by flagging them or manually changing them.  

## TODO
We have accumulated the following to-do list, which you can expect to be done in the very near future
- Train SSD300 with batch norm (in progress)
- Add support for COCO dataset
- Add support for SSD512 training and testing
- Create a functional model definition for Sergey Zagoruyko's [functional-zoo](https://github.com/szagoruyko/functional-zoo) (in progress)

## Comments
Please feel-free to post an issue with any questions or suggestions.  We tried to keep this implementation as pure to PyTorch as
possible so as to 1) learn the framework and 2) give back to the [PyTorch](http://pytorch.org/) community.  That being said, there are ports of the original [SSD](https://github.com/weiliu89/caffe/tree/ssd) to other frameworks now, so some of those were taken into consideration when optimizing our code.  We would also like to acknowledge Soumith Chintala and Adam Paszke for their dedication and support on the
[PyTorch Discussion Page](https://discuss.pytorch.org/), as well as Jimmy Whitaker for introducing me to the framework and to this paper.


## References
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [Original Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- A list of other great SSD ports that were sources of inspiration: 
  * [Keras](https://github.com/rykov8/ssd_keras), [Chainer](https://github.com/Hakuyume/chainer-ssd), [MXNet](https://github.com/zhreshold/mxnet-ssd), [Tensorflow](https://github.com/balancap/SSD-Tensorflow) 
