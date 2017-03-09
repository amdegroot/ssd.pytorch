import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
from box_utils import *

class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.

    """
    def __init__(self, num_classes, background_label, keep_top_k, conf_thresh, nms_threshold, nms_top_k):
        #super(Detect, self).__init__()
        self.num_classes = num_classes
        self.background_label = background_label
        self.keep_top_k = keep_top_k
        # Parameters used in nms.
        self.nms_threshold = nms_threshold
        if nms_threshold <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.top_k = -1
        self.nms_top_k = nms_top_k or -1
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
                Shape: [1,2,num_priors*4]
        """
        num = loc_data.size(0) # batch size
        self.output = torch.Tensor(num,self.keep_top_k,7)  # TODO: refactor

        # num_priors = prior_data.size(0) // 4  # height dim of priorbox input / 4
        # assert(loc_data.size(1) == prior_data.size(0))
            # raise ValueError('Number of priors must match number of location predictions.')

        # GET CONFIDENCE SCORES FROM CONF_DATA
        # If input is only a single image then we need to add the batch dim
        # that we removed for softmax layer
        print(prior_data.size())
        print(loc_data.size())
        if num == 1:
            conf_preds = conf_data.t().contiguous().unsqueeze(0) # size num x 21 x 7308
        else:
            conf_preds = conf_data.view(num,num_priors,self.num_classes).transpose(2,1)

        # GET PRIOR BBOXES FROM PRIOR_DATA
        # prior_bboxes = prior_data[0][0].view(-1,4)    # Shape [7308 x 4]
        # prior_variances = prior_data[0][1].view(-1,4) # Shape [7308 x 4]
        prior_variances = torch.Tensor([0.1,0.1,0.2,0.2])
        #TODO get rid of the batch for loop
        # Decode predictions into bboxes.
        num_kept = 0
        for i in range(num):
            decode_bboxes = decode_boxes(prior_data,prior_variances,loc_data[i])
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            indices = []
            num_det = 0
            for c in range(self.num_classes):
                if not c == self.background_label:
                    # idx of highest scoring and non-overlapping boxes for a given class
                    indices.append(nms(decode_bboxes, conf_scores[c], self.nms_threshold, self.top_k))
                    # Class_index_table now contains
                    # TODO optimize
                    # indices = (num_classes X num_det(which could vary over classes,
                    # so using table over tensor))
                    num_det += indices[len(indices)-1].size(0)
            length = num_det  # length of tensors to be created based on num boxes after nms
            score_pairs = torch.Tensor(length)  # scores and corresponding bbox table indices
            indices_list = torch.Tensor(length)
            label_list = torch.Tensor(length)
            ref = 0
            for number in range(self.num_classes-1):
                label = number + 1 # add 1 to avoid background class
                label_indices = indices[number]  # top bbox table indices for each class
                for index in range(label_indices.size(0)):
                    idx = int(label_indices[index])
                    # inserting index of highest conf scores into indices list
                    indices_list[ref] = idx
                    assert(idx <= conf_scores[label].size(0))
                    # corresp. score is inserted into score_pairs at same location
                    score_pairs[ref] = conf_scores[label][idx]
                    # label is added to label list at same location
                    label_list[ref] = label
                    ref +=1
            if num_det > self.keep_top_k: # narrow results further
                length = self.keep_top_k
            final_indices = torch.Tensor(length).zero_()
            final_scores = torch.Tensor(length).zero_()
            final_labels = torch.Tensor(length).zero_()
            sort(score_pairs, indices_list, label_list, length, final_scores, final_indices, final_labels)
            num_kept += num_det

            for j in range(final_indices.size(0)):
                idx = int(final_indices[j])
                self.output[i][j][0] = i+1
                self.output[i][j][1] = final_labels[j]
                self.output[i][j][2] = final_scores[j]
                self.output[i][j][3] = max(min(decode_bboxes[idx][0], 1), 0)
                self.output[i][j][4] = max(min(decode_bboxes[idx][1], 1), 0)
                self.output[i][j][5] = max(min(decode_bboxes[idx][2], 1), 0)
                self.output[i][j][6] = max(min(decode_bboxes[idx][3], 1), 0)
        return self.output
