

class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the np.ndarray, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        if random.randrange(2):
            tmp = image[:, :, 1].astype(float) * random.uniform(self.lower, self.upper)
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            image[:, :, 1] = tmp
        return image


class RandomHue(object):
    def __init__(self, delta=18):
        self.delta = delta

    def __call__(self, image):
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-self.delta, self.delta)
            tmp %= 180
            image[:, :, 0] = tmp
        return image


class RandomLightingNoise(object):
    def __init__(self, n_channels):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image):
        if random.randrange(2):
            swap = random.choice(self.perms)
            shuffle = SwapChannel(swap)  # shuffle channels
            image = shuffle(image)
        return image


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform

    def __call__(self, image):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        if random.randrange(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
            image[image < 0] = 0
            image[image > 255] = 255
        return image


class RandomBrightness(object):
    def __init__(self, delta=32):
        self.delta = delta
        assert self.delta >= 0, "brightness delta must be non-negative."

    def __call__(self, image):
        if random.randrange(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
            image[image < 0] = 0
            image[image > 255] = 255
        return image

class ToCV2Image(object):
    def __call__(self, tensor):
        # may have to call cv2.cvtColor() to get to BGR
        return tensor.cpu().numpy().astype(np.float32)

class ToTensor(object):
    def __call__(self, cvimage):
        return torch.from_numpy(cvimage)


class RandomSampleCrop(img, boxes, labels, mode):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __call__(img, boxes, labels, mode):
        while True:
            height, width, _ = image.shape

            if mode is None:
                return img, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            for _ in range(50):
                w = random.randrange(int(0.3 * width), width)
                h = random.randrange(int(0.3 * height), height)

                # aspect ratio b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue
                left = random.randrange(width - w)
                top = random.randrange(height - h)
                rect = torch.LongTensor([[left, top, left + w, top + h]])
                overlap = jaccard(boxes, rect)
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue
                # t = transforms.ToTensor()
                # p = transforms.ToPILImage()
                image = p((img)[:, rect[0, 1]:rect[0, 3], rect[0, 0]:rect[0, 2]])

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                m1 = (rect[0, :2].expand_as(centers).lt(centers)).sum(1).gt(1)
                m2 = (centers.lt(rect[0, 2:].expand_as(centers))).sum(1).gt(1)
                mask = (m1 + m2).gt(1).squeeze().nonzero().squeeze()

                if mask.dim() == 0:
                    continue
                boxes = boxes[mask].clone()
                classes = labels[mask]
                rect = rect.expand_as(boxes)
                boxes[:, :2] = torch.max(boxes[:, :2], rect[:, :2])
                boxes[:, :2] -= rect[:, :2]
                boxes[:, 2:] = torch.min(boxes[:, 2:], rect[:, 2:])
                boxes[:, 2:] -= rect[:, 2:]

                return image, boxes, classes

class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randrange(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels as specified in the swap
    modifies the input tensor
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image.transpose(*swaps)
        return image
