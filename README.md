# ssd.pytorch
A PyTorch implementation of [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325)

## Contents
1. [Requirements](#requirements)
2. [Training](#a.-training-ssd)
3. [Download Dataset](#dataset-download)
4. [Demo](#b.-Use-a-pre-trained-SSD-network-for-detection)
5. [Test/Eval](#c.-test/evaluate)
6. [Comments](#comments)
7. [Reference](#references)


## Requirements
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website. 
- `pip install -r requirements.txt`
- Clone this repo 
- Download the PASCAL VOC Dataset via provided shell scripts

# A. Training SSD
- First download the [VGG-16](https://arxiv.org/abs/1409.1556) base network weights which can be found [here](https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth), courtesy of [Justin Johnson](https://github.com/jcjohnson/pytorch-vgg).
- Then download the dataset by following the instructions below. 
- Note: Currently we only support [VOC](http://host.robots.ox.ac.uk/pascal/VOC/), but are adding [COCO](http://mscoco.org/) and hopefully [ImageNet](http://www.image-net.org/) soon. 
## Dataset Download

### VOC Dataset
##### VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```
##### VOC2012 trainval
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

### Train Script 
- To train SSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.
- For training, an NVIDIA GPU is strongly recommended for speed. 
```Shell 
python train.py
```
 
# B. Use a pre-trained SSD network for detection

## Download a pre-trained network
A pretrained SSD (pool6/non-bn version) can be found here
- https://s3.amazonaws.com/amdegroot-models/ssd_300_voc07.tar.gz

## Try the demo
- Make sure you have [jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html) installed. 
- Two alternatives for installing jupyter notebook
- If you installed PyTorch with [conda](https://www.continuum.io/downloads) (recommended), then you should already have it.  Just navigate to the ssd.pytorch cloned repo and run:
```Shell
jupyter notebook
```
- If using [pip](https://pypi.python.org/pypi/pip) package manager:
```Shell
# make sure pip is upgraded 
pip3 install --upgrade pip 

# install jupyter notebook
pip install jupyter

# Run this inside ssd.pytorch
jupyter notebook 
```
- Now navigate to `demo.ipynb` in the browser window that pops up and have at it!
- This can be done on both CPU and GPU. 

# C. Test/Evaluate
To evaluate a trained network:
```Shell
python test.py
```
You can specify the parameters listed in the `test.py` file by flagging them or manually changing them.  

## TODO 
We have accumulated the following to-do list, which you can expect to be done in the very near future:
1. Train SSD300 with batch norm 
2. Add support for COCO dataset 
3. Add support for SSD512 training and testing
4. Create a functional model definition for Sergey Zagoruyko's [functional-zoo](https://github.com/szagoruyko/functional-zoo)

## Comments
Please feel-free to post an issue with any questions or suggestions.  We tried to keep this implementation as pure to PyTorch as 
possible so as to 1) learn the framework and 2) give back to the [PyTorch](http://pytorch.org/) community.  That being said, there are ports of the original [SSD](https://github.com/weiliu89/caffe/tree/ssd) to other frameworks now, so some of those were taken into consideration when optimizing our code.  We would also like to acknowledge Soumith Chintala and Adam Paszke for their dedication and support on the 
[PyTorch Discussion Page](https://discuss.pytorch.org/), as well as Jimmy Whitaker for introducing me to the framework and to this paper. 


## References
- Wei Liu, et al. "SSD: Single shot multibox detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [Original Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- Justin Johnson, [jcjohnson/pytorch-vgg](https://github.com/jcjohnson/pytorch-vgg)
