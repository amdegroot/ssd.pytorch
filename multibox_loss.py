import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
from utils import *

class MultiBoxLoss(nn.Module):
    def __init__(num_classes_,match_type,overlap_thresh,prior_for_matching,background_label_,neg_mining,neg_pos,neg_overlap,code_type_,labels):
        self.num_classes = num_classes_
        self.match_type = match_type
        self.overlap_threshold = overlap_thresh
        self.num_loc_classes = num_classes_
        self.background_label = background_label_
        self.code_type = code_type_
        self.encode_in_target = var_encoded_in_target_
        self.use_prior_for_matching  = prior_for_matching
        self.do_neg_mining = neg_mining
        self.neg_pos_ratio = neg_pos
        self.neg_overlap = neg_overlap_


    def forward(self, loc_data, conf_data, prior_data, ground_truth):
        self.loc_data = loc_data # batch_size X num_priors*4
        self.conf_data = conf_data # batch_size*num_priors X num_classes
        self.prior_data = prior_data # 1 X 2 X num_priors*4
        num = self.loc_data.size(0)
        self.output = []
        num_priors = (self.prior_data.size(2)) // 4
        num_gt = self.gt_data.size(0)
        num_classes = self.num_classes
        num = self.num
        max_scores = max_conf(num, self.conf_data, self.num_classes, self.background_label)
        num_matches = 0
        num_negs = 0

        # for i in range(num):
        #     if len(all_gt_bboxes[i]) <= 0: # Check if there is ground truth for current image
        #         continue
        # Find match between predictions and ground truth
        truths = ground_truths[i]
        location_targets, class_targets = match(truths, priors, variances, labels, threshold)
        pos = class_targets > 0  # [N,8732] pos means the box matched.
        num_pos = pos.sum()
        self.num_matches += num_pos

        pos_mask = pos.unsqueeze(2).expand_as(loc_preds)    # [N,8732,4]
        pos_loc_preds = loc_preds[pos_mask].view(-4)  # [#pos,4]
        pos_loc_targets = loc_targets[pos_mask].view(-1,4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(pos_loc_preds, pos_loc_targets, size_average=False)

        conf_preds = conf_preds.view(-1,self.num_classes) #x
        conf_targets.view(-1) #y

        max_conf = torch.max(conf_preds)
        log_sum_exp = torch.log(torch.sum(torch.exp(conf_preds-max_conf), 1)) + max_conf
        conf_loss = log_sum_exp - conf_preds.gather(1, conf_targets.view(-1,1))


        batch_size = pos.size(0)
        num_boxes = pos.size(1)

        conf_loss[pos] = 0  # set pos boxes = 0, the rest are neg conf_loss
        conf_loss = conf_loss.view(batch_size, -1)  # [N,8732]
        max_loss,_ = conf_loss.sort(1, descending=True)  # sort by neg conf_loss

        num_pos = pos.long().sum(1)  # [N,1]
        num_neg = torch.clamp(3*num_pos, max=num_boxes-1)  # [N,1]

        pivot_loss = max_loss.gather(1, num_neg)           # [N,1]
        neg = conf_loss > pivot_loss.expand_as(conf_loss)  # [N,8732]

        pos_mask = pos.unsqueeze(2).expand_as(conf_preds)  # [N,8732,21]
        neg_mask = neg.unsqueeze(2).expand_as(conf_preds)  # [N,8732,21]
        mask = (pos_mask+neg_mask).gt(0)

        pos_and_neg = (pos+neg).gt(0)
        preds = conf_preds[mask].view(-1,self.num_classes)  # [#pos+#neg,21]
        targets = conf_targets[pos_and_neg]                 # [#pos+#neg,]
        conf_loss = F.cross_entropy(preds, targets, size_average=False)
        loss = (loc_loss + conf_loss) / num_matched_boxes
        return loss
