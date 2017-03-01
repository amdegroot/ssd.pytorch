import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
from utils import *

class Detect(Function):
    def __init__(self, num_classes_, share_location_, background_label_id_, code_type_, var_encoded_in_target_, keep_top_k_, conf_thresh, nms_threshold_, nms_top_k_, visualize_, vis_thresh_):
        #super(Detect, self).__init__()
        # initializations ........
        self.num_classes = num_classes_
        self.share_location = share_location_ or True
        if share_location_:
            self.num_loc_classes = 1
        else:
            self.num_loc_classes = num_classes_
        self.background_label_id = background_label_id_
        self.code_type = code_type_
        self.variance_encoded_in_target = var_encoded_in_target_
        self.keep_top_k = keep_top_k_

        # Parameters used in nms.
        self.nms_threshold = nms_threshold_
        if nms_threshold_ <= 0:
            error('<Detection_Output> nms_threshold must be non negative.')
        self.top_k = -1
        self.nms_top_k = nms_top_k_ or -1
        if self.nms_top_k > 0:
            self.top_k = self.nms_top_k

        self.visualize = visualize_
        self.vis_thresh = vis_thresh_ or 0.6
        # reset the data transformer (randomly re-initialize it)

    def forward(self, loc_data, conf_data, box_data):

        self.loc_data = loc_data # batch_size X num_priors*4
        self.conf_data = conf_data # batch_size*num_priors X num_classes
        self.prior_data = box_data # 1 X 2 X num_priors*4
        num = self.loc_data.size(0)
        self.output = torch.Tensor(num,self.keep_top_k,7) # could be some empty indices, but this way we output 1 tensor

        num_priors = self.prior_data.size(2) // 4      # height dim of priorbox input / 4
        if not self.loc_data.size(1) == self.prior_data.size(2):
            print('Number of priors must match number of location predictions.')


        # GET LOCATION PREDICTIONS FROM LOC_DATA
        # (num X num_priors X 4) -- in caffe
        # (right now num X (num_priors x 4))
        #print(num)
        loc_preds = self.loc_data

        # GET CONFIDENCE SCORES FROM CONF_DATA

        if num == 1: # If input is only a single image then we need to add the batch dim that we removed for softmax layer
            conf_preds = self.conf_data.t().contiguous()
            conf_preds = conf_preds.unsqueeze(0) # size num x 21 x 7308
            #print(conf_preds.size())
        else:
            conf_predictions = self.conf_data.view(num,num_priors,self.num_classes)
            conf_preds = conf_predictions.transpose(2,1)

        # GET PRIOR BBOXES FROM PRIOR_DATA  -- Still thinking of good way to speed this up

        prior_bboxes = self.prior_data[0][0].view(-1,4)  # size 7308 x 4
        prior_variances = self.prior_data[0][1].view(-1,4) # size 7308 x 4
        #print(prior_bboxes.size())
        # Decode predictions into bboxes.
        num_kept = 0
        for i in range(num):
            decode_bboxes =  decode_boxes(prior_bboxes,prior_variances,self.code_type,self.variance_encoded_in_target,loc_preds[i])
            # For each class, perform nms
            conf_scores = conf_preds[i]

            indices = []
            num_det = 0
            print(self.num_classes)
            print(self.nms_threshold)
            print(self.top_k)
            for c in range(self.num_classes):
                if not (c == self.background_label_id):
                    #  Populates overlaps and class_index_table
                    class_index_table = apply_nms(decode_bboxes, conf_scores[c].clone(), self.nms_threshold, self.top_k)
                    # Class_index_table now contains the indices (with respect to num_priors) of highest scoring and non-overlapping bboxes for a given class
                    indices.append(class_index_table)
                    # indices = (num_classes X num_det(which could vary over classes, thus using table over tensor))
                    num_det += class_index_table.size(0)

            length = num_det  # length of tensors to be created based on num boxes after nms
            score_pairs = torch.Tensor(length)  # will score scores and corresponding bbox table indices
            indices_list = torch.Tensor(length)
            label_list = torch.Tensor(length)
            ref = 0
            for number in range(self.num_classes-1):
                label = number + 1 # add 1 to avoid background class
                label_indices = indices[number]  # top bbox table indices for each class
                for index in range(label_indices.size(0)):


                    idx = int(label_indices[index])
                    indices_list[ref] = idx  #  inserting index of highest conf scores into indices list
                    assert(idx <= conf_scores[label].size(0))

                    score_pairs[ref] = conf_scores[label][idx]  #  corresp. score is inserted into score_pairs at same location
                    label_list[ref] = label  # and label is added to label list at same location
                    #  score_pairs, indices_list, and label_list could be combined into one tensor potentially
                    ref +=1


            if num_det > self.keep_top_k: # narrow results further
                length = self.keep_top_k

            final_indices = torch.Tensor(length).zero_()
            final_scores = torch.Tensor(length).zero_()
            final_labels = torch.Tensor(length).zero_()
            sort(score_pairs, indices_list, label_list, length, final_scores, final_indices, final_labels)
            num_kept += num_det

            # print(final_indices.size())
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
