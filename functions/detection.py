import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
from utils.box_utils import decode, nms


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.

    """

    def __init__(self, num_classes, background_label, keep_top_k, conf_thresh, nms_threshold, nms_top_k):
        self.num_classes = num_classes
        self.background_label = background_label
        self.keep_top_k = keep_top_k
        # Parameters used in nms.
        self.nms_threshold = nms_threshold
        if nms_threshold <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.top_k = -1
        self.nms_top_k = nms_top_k or -1
        self.conf_thresh = conf_thresh
        if self.nms_top_k > 0:
            self.top_k = self.nms_top_k

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers,
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers,
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers,
                Shape: [1,num_priors,4]
        """

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(1)
        self.output = torch.zeros(
            num, self.num_classes, self.keep_top_k, 5)  # TODO: refactor
        if num == 1:
            conf_preds = conf_data.t().contiguous().unsqueeze(0)  # size num x 21 x 7308
        else:
            conf_preds = conf_data.view(
                num, num_priors, self.num_classes).transpose(2, 1)
        variances = torch.Tensor([0.1, 0.1, 0.2, 0.2])
        # Decode predictions into bboxes.
        num_kept = 0
        for it in range(num):
            decoded_boxes = decode(loc_data[it], prior_data, variances)
            # For each class, perform nms
            conf_scores = conf_preds[it].clone()
            num_det = 0
            for cl in range(1, self.num_classes):
                # idx of highest scoring and non-overlapping boxes for a given
                # class
                score_points, count = nms(decoded_boxes, conf_scores[cl],
                                          self.nms_threshold, self.keep_top_k)
                self.output[it, cl, :count] = score_points
        return self.output
