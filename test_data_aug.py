import cv2
from utils import augmentations as aug
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from data import VOCDetection, AnnotationTransform
img = cv2.imread('/Users/alex/Downloads/VOCdevkit/VOC2012/JPEGImages/2012_004284.jpg')

dataset = VOCDetection('/Users/alex/Downloads/VOCdevkit', image_sets=[('2007', 'train')], dataset_name='VOC2007', transform=aug.SSDAugmentation(), target_transform=AnnotationTransform())

img, target = dataset[100]
print(target)

boxes = target[:, :4].numpy()


channels, height, width = img.size()

# adjust boxes spect to image dimensions
boxes[:, 0] *= width
boxes[:, 2] *= width
boxes[:, 1] *= height
boxes[:, 3] *= height

# Create figure and axes
fig, ax = plt.subplots(1)

disp_image = img.numpy().transpose((1, 2, 0))[:, :, ::-1]
disp_image -= disp_image.min()
disp_image /= disp_image.max()

ax.imshow(disp_image)

for box in boxes:
    print('HAVE BOXES')
    print('B:', box)
    # print(box[0], box[1], box[2], box[3])
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0]-1, box[3]-box[1]-1, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

plt.show()
