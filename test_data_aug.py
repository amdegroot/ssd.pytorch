import cv2
from utils import augmentations as aug
# from utils import augment as aug
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from data import VOCDetection, AnnotationTransform
img = cv2.imread('/Users/alex/Downloads/VOCdevkit/VOC2012/JPEGImages/2012_004284.jpg')

dataset = VOCDetection('/Users/alex/Downloads/VOCdevkit', image_sets=[('2007', 'train')], dataset_name='VOC2007', target_transform=AnnotationTransform())

img, target = dataset[100]
img = img.astype(np.float32)/256.0
target = np.array(target)

means=(104/256.0, 117/256.0, 123/256.0)
# t = aug.SSDAugmentation()
# t = aug.Expand(means)
t = aug.RandomSampleCrop()

boxes = target[:, :4]
height, width, channels = img.shape
boxes[:, 0] *= width
boxes[:, 2] *= width
boxes[:, 1] *= height
boxes[:, 3] *= height

print(boxes)

labels = target[:, 4]
timg, tboxes, tlabels = t(img, boxes, labels)

print(timg.shape)

# Create figure and axes
fig,ax = plt.subplots(1)

ax.imshow(timg[:, :, ::-1])

for box in tboxes:
    print('HAVE BOXES')
    print('B:', box)
    # print(box[0], box[1], box[2], box[3])
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

plt.show()
