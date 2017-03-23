import cv2
import numpy as np
import random

def crop(image, boxes, classes):
    height, width, _ = image.shape

    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, classes

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            w = random.randrange(int(0.3 * width), width)
            h = random.randrange(int(0.3 * height), height)

            if h / w < 0.5 or 2 < h / w:
                continue

            rect = torch.LongTensor([[random.randrange(width - w),
                                    random.randrange(height - h),
                                    w,h]])

            overlap = jaccard(boxes, rect)
            if overlap.min() < min_iou and max_iou < overlap.max():
                continue

            image = image[rect[0,1]:rect[0,3], rect[0,0]:rect[0,2]]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            m1 = rect[0,:2].expand_as(centers) < centers
            m2 = centers < rect[0,2:].expand_as(centers)
            mask = (m1+m2).gt(1) # equivalent to logical-and

            boxes = boxes[mask].copy()
            classes = classes[mask]
            boxes[:, :2] = torch.max(boxes[:, :2], rect[:,:2].expand_as(boxes))
            boxes[:, :2] -= rect[:,:2].expand_as(boxes)
            boxes[:, 2:] = torch.min(boxes[:, 2:], rect[:,2:].expand_as(boxes))
            boxes[:, 2:] -= rect[:,2:].expand_as(boxes)

            return image, boxes, classes


def distort(image, boxes, classes):
    def convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image, boxes, classes


def expand(image, boxes, classes, mean):
    if random.randrange(2):
        return image, boxes, classes

    height, width, depth = image.shape
    ratio = random.uniform(1, 4)
    left = random.randint(0, int(width * ratio) - width)
    top = random.randint(0, int(height * ratio) - height)

    expand_image = np.empty(
        (int(height * ratio), int(width * ratio), depth),
        dtype=image.dtype)
    expand_image[:, :] = mean
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    boxes = boxes.copy()
    boxes[:, :2] += torch.Tensor(left,top)
    boxes[:, 2:] += torch.Tensor(left,top)

    return image, boxes, classes


def mirror(image, boxes, classes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes, classes


def augment(image, boxes, classes, mean):
    image, boxes, classes = crop(image, boxes, classes)
    image, boxes, classes = distort(image, boxes, classes)
    image, boxes, classes = expand(image, boxes, classes, mean)
    image, boxes, classes = mirror(image, boxes, classes)
    return image, boxes, classes


class AnnotatedDataLayer():

    def __init__(self, dataset, size, mean, encoder):
        super().__init__()

        self.dataset = dataset
        self.size = size
        self.mean = mean

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        image =

        try:
            boxes, classes = zip(*self.dataset.annotations(i))
            boxes = np.array(boxes)
            classes = np.array(classes)
            image, boxes, classes = augment(image, boxes, classes, self.mean)
        except ValueError:
            boxes = np.empty((0, 4), dtype=np.float32)
            classes = np.empty((0,), dtype=np.int32)

        h, w, _ = image.shape
        image = cv2.resize(image, (self.size, self.size)).astype(np.float32)
        image -= self.mean
        image = image.transpose(2, 0, 1)
        boxes[:, 0::2] /= w
        boxes[:, 1::2] /= h
        loc, conf = self.encoder.encode(boxes, classes)

        return image, loc, confz
