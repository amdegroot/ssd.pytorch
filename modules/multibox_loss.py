import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from box_utils import *
cud = False
if torch.cuda.is_available():
    cud = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

class MultiBoxLoss(nn.Module):
    def __init__(self,num_classes,overlap_thresh,prior_for_matching,bkg_label,neg_mining,neg_pos,neg_overlap,encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching  = prior_for_matching
        self.do_neg_mining = neg_mining
        self.neg_pos_ratio = neg_pos
        self.neg_overlap = neg_overlap


    def forward(self, predictions, ground_truth):
        loc_data, conf_data, priors = predictions

        num = loc_data.size(0)
        num_priors = (priors.size(1)) // 4
        num_classes = self.num_classes
        #max_scores = max_conf(num, self.conf_data, self.num_classes, self.background_label)
        num_matches = 0
        num_negs = 0
        loc_targets = torch.Tensor(num, num_priors, 4)
        conf_targets = torch.LongTensor(num, num_priors)
        for i in range(num):
            truths = ground_truth[i][:,:-1].data
            labels = ground_truth[i][:,-1].data
            priorboxes = priors[0].view(-1,4).data
            variances = priors[1].view(-1,4).data
            match(self.threshold,truths,priorboxes,variances,labels,loc_targets,conf_targets,i)

        if cud:
            loc_targets = loc_targets.cuda()
            conf_targets = conf_targets.cuda()
        loc_targets = Variable(loc_targets)
        conf_targets = Variable(conf_targets)
        pos = conf_targets > 0  # [N,8732] pos means the box matched.
        num_pos = pos.sum()
        num_matches += num_pos

        pos_mask = pos.unsqueeze(pos.dim()).expand_as(loc_data.squeeze())    # [N,8732,4]

        pos_loc_preds = loc_data[pos_mask].view(-1,4)  # [num_pos,4]

        pos_loc_targets = loc_targets[pos_mask].view(-1,4)  # [num_pos,4]

        loc_loss = F.smooth_l1_loss(pos_loc_preds, pos_loc_targets, size_average=False)

        conf_data_temp = conf_data.view(-1,self.num_classes) #x

        max_conf = conf_data.data.max()

        log_sum_exp = torch.log(torch.sum(torch.exp(conf_data_temp-max_conf), 1)) + max_conf
        conf_loss = log_sum_exp - conf_data_temp.gather(1, conf_targets.view(-1,1))

        batch_size = pos.size(0)
        num_boxes = pos.size(1)


        conf_loss[pos] = 0  # set pos boxes = 0, the rest are neg conf_loss
        conf_loss = conf_loss.view(batch_size, -1)  # [N,8732]
        _,loss = conf_loss.sort(1, descending=True)  # sort by neg conf_loss
        _,rank = loss.sort(1)

        num_pos = pos.long().sum(1)  # [N,1]
        num_neg = torch.clamp(3*num_pos, max=num_boxes-1)  # [N,1]

        neg = rank < num_neg.expand_as(rank)


        pos_mask = pos.unsqueeze(2).expand_as(conf_data)  # [N,8732,21]
        neg_mask = neg.unsqueeze(2).expand_as(conf_data)  # [N,8732,21]
        mask = (pos_mask+neg_mask).gt(0)

        pos_and_neg = (pos+neg).gt(0)

        preds = conf_data[mask].view(-1,self.num_classes)  # [num_pos+num_neg,num_classes]

        targets = conf_targets[pos_and_neg]                 # [num_pos+num_neg,]

        conf_loss = F.cross_entropy(preds, targets, size_average=False)
        loss = (loc_loss + conf_loss) / num_pos.data.sum()
        return loss
