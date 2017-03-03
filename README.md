# ssd.pytorch
A PyTorch implementation of [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325)


## Preparation

### Download the VOC2007 dataset. We assume the data is stored in `~/data/`

```Shell
# Download the data.
cd ~/data
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# Extract data
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
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