# ssd.pytorch
A PyTorch implementation of [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325)

## Contents
1. [Requirements](#requirements)
2. [Download](#dataset-download)
3. [Reference](#references)


## Requirements
- Install PyTorch ([pytorch.org](http://pytorch.org/))
- `pip install -r requirements.txt`
- Download the PASCAL VOC Dataset via provided shell scripts

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



## References
- Wei Liu, et al. "SSD: Single shot multibox detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [Original Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)