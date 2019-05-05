import torch
from torch.autograd import Function
from ..box_utils import decode
from data import voc as cfg

class paper_box(object):
    def __init__(self, index, x, y, box):
        self.index = index
        self.x = x
        self.y = y
        self.box = box
def box_iou(a, b):
    if a.box[2] < b.box[0] or a.box[0] > b.box[2]:
        return 0
    if a.box[1] > b.box[3] or a.box[3] < b.box[1]:
        return 0
    width = min(a.box[2], b.box[2]) - max(a.box[0], b.box[0])
    height = min(a.box[3], b.box[3]) - max(a.box[1], a.box[1])
    iou = width * height
    a_area = (a.box[2] - a.box[0]) * (a.box[3] - a.box[1])
    b_area = (b.box[2] - b.box[0]) * (b.box[3] - b.box[1])
    return (iou / (a_area + b_area - iou))

class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

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
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)
        #next we will specific the exact layer and its output
        #we get the all predicted boxes and its confidence
        decoded_boxes = decode(loc_data[0], prior_data, self.variance)
        conf_data = conf_data[0]
        loc_data = loc_data[0]
        all_boxes = torch.cat((decoded_boxes, conf_data), 1)
#        for i in range(self.num_classes):
#            index = []
#            for j in range(len(loc_data)):
#                index.append(j)
#            #in the specific class, we will do something specifical
#           for j in range(len(loc_data)):
#                for k in range(len(loc_data) - j):
#                    if conf_data[j][i] < conf_data[k][i]:
#                        index[j] = 
        return all_boxes

        # Decode predictions into bboxes.
#        for i in range(num):
#            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
#            # For each class, perform nms
#            conf_scores = conf_preds[i].clone()
#
#            for cl in range(1, self.num_classes):
#                c_mask = conf_scores[cl].gt(self.conf_thresh)
#                scores = conf_scores[cl][c_mask]
#                if scores.size(0) == 0:
#                    continue
#                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
#                boxes = decoded_boxes[l_mask].view(-1, 4)
#                # idx of highest scoring and non-overlapping boxes per class
#                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
#                output[i, cl, :count] = \
#                    torch.cat((scores[ids[:count]].unsqueeze(1),
#                               boxes[ids[:count]]), 1)
#        flt = output.contiguous().view(num, -1, 5)
#        _, idx = flt[:, :, 0].sort(1, descending=True)
#        _, rank = idx.sort(1)
#        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
#        return output
