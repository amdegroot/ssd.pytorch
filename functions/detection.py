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
    def __init__(self, num_classes, background_label, keep_top_k, conf_thresh,
                 nms_threshold, nms_top_k):
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

    def conf_tresh(self, locs, scores, priors, thresh=0.01):
        score_filter = scores.gt(thresh)

        scores = scores[score_filter]
        return scores[score_filter], locs[score_filter], priors[score_filter]

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        self.output = torch.zeros(num, self.num_classes, self.keep_top_k, 5)
        if num == 1:
            # size batch x num_classes x 7308
            conf_preds = conf_data.t().contiguous().unsqueeze(0)
        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes).transpose(2, 1)
        variances = torch.Tensor([0.1, 0.1, 0.2, 0.2])
        # Decode predictions into bboxes.
        score_filter = torch.ByteTensor(conf_preds.size(2))
        score_ids = torch.range(0,conf_preds.size(2)-1).long()
        for i in range(num):
            # loc_data, conf_preds, prior_data = conf_thresh(loc_data[i], conf_preds[i], prior_data[i])
            decoded_boxes = decode(loc_data[i], prior_data, variances)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            num_det = 0
            for c in range(1, self.num_classes):
                c_mask = conf_scores[c].ge(self.conf_thresh)
                # print(mask.size())
                # ids = score_ids[c_mask]
                scores = conf_scores[c][c_mask]
                if scores.dim() == 0:
                    continue
                # if scores.size
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1,4)
                # boxes = torch.gather(decoded_boxes, 0, ids)
                # scores = torch.gather(conf_scores[c], 0, ids)
                # print(boxes.size())
                # print(scores.size())
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores,
                                           self.nms_threshold, self.keep_top_k)
                # score_points, count = nms(boxes, scores,
                #                           self.nms_threshold, self.keep_top_k)
                # if score_points.dim()==0:
                #     continue
                # print(score_points.size())
                # print(self.output[i, c, :10].size())
                self.output[i, c, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                  boxes[ids[:count]]), 1)
                self.output[i, c, 10:].fill_(0)

        return self.output
