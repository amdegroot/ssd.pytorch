import torch
import torch.nn as nn
from torch.autograd import Function
from utils import *

class MultiBox(Function):
    def __init__(num_classes_,share_location_,match_type,overlap_thresh,prior_for_matching,background_label_id_,difficult_gt,neg_mining,neg_pos,neg_overlap,code_type_,var_encoded_in_target_,labels):

        self.num_classes = num_classes_
        self.match_type = match_type
        self.overlap_threshold = overlap_thresh
        self.share_location = share_location_ or True
        if share_location_:
            self.num_loc_classes = 1
        else
            self.num_loc_classes = num_classes_

        self.background_label_id = background_label_id_
        self.code_type = code_type_
        self.encode_in_target = var_encoded_in_target_

        self.use_prior_for_matching  = prior_for_matching
        self.use_difficult_gt = difficult_gt
        self.do_neg_mining = neg_mining
        self.neg_pos_ratio = neg_pos
        self.neg_overlap = neg_overlap_
        # self.labels = labels or torch.CudaTensor()
        # self.output = []


    def forward(self, loc_data, conf_data, prior_data, ground_truth):

        self.loc_data = loc_data # batch_size X num_priors*4
        self.conf_data = conf_data # batch_size*num_priors X num_classes
        self.prior_data = prior_data # 1 X 2 X num_priors*4
        self.gt_data = ground_truth
        num = self.loc_data.size(0)


        self.output = []
        num_priors = (self.prior_data.size(2)) // 4
        num_gt = self.gt_data.size(0)
        num_classes = self.num_classes
        num = self.num

      # GET ALL GROUND TRUTH
        all_gt_bboxes = []
        for x in range(self.loc_data.size(0)):
            all_gt_bboxes[x] = []

        getGroundTruth(self.gt_data, num_gt, self.num, self.background_label_id, self.use_difficult_gt, all_gt_bboxes)

        # GET PRIOR BBOXES FROM PRIOR_DATA
        prior_bboxes = []
        prior_variances = []
        getPriorBboxes(self.prior_data, num_priors, prior_bboxes, prior_variances)

        # GET LOCATION PREDICTIONS FROM LOC_DATA

        # GET MAX CONFIDENCE SCORES FOR EACH PRIOR. USED IN NEGATIVE MINING.
        all_max_scores = torch.Tensor()
        getMaxConfScores(self.conf_data.clone(), num, self.num_classes, self.background_label_id, all_max_scores)

        self.all_match_indices = []
        self.all_neg_indices = []
        self.num_matches = 0
        num_negs = 0

        for i in range(num):
            match_indices = torch.Tensor()
            neg_indices = torch.Tensor()
            if len(all_gt_bboxes[i]) <= 0: # Check if there is ground truth for current image
                self.all_match_indices.append(match_indices)
                self.all_neg_indices.append(neg_indices)
                continue
            # Find match between predictions and ground truth
            gt_bboxes = all_gt_bboxes[i]
            match_overlaps = torch.Tensor()
            temp_match_indices = torch.Tensor(len(prior_bboxes)).fill_(-1)
            temp_match_overlaps = torch.Tensor(len(prior_bboxes)).fill_(0)
            label = -1
            # matchBbox(gt_bboxes,prior_bboxes,label,self.match_type,self.overlap_threshold,temp_match_indices,temp_match_overlaps)
            matchBbox(gt_bboxes,prior_bboxes,label,self.match_type,self.overlap_threshold)
            match_indices.resize_as_(temp_match_indices).copy_(temp_match_indices)
            match_overlaps.resize_as_(temp_match_overlaps).copy_(temp_match_overlaps)


            num_pos = temp_match_indices[temp_match_indices.ne_(-1)].size(0)
            self.num_matches += num_pos
            if self.do_neg_mining:
                num_neg = 0
                # filtering the indices of the negative examples (those less than our overlap threshold)
                indices = torch.range(1,temp_match_overlaps.nelement())[temp_match_overlaps.lt_(self.overlap_threshold)]
                values = temp_match_overlaps[temp_match_overlaps.lt_(self.overlap_threshold)] #their corresponding values (overlaps)
                score_tensor = torch.Tensor(indices.size(0)).zero_()
                index_tensor = torch.Tensor(indices.size(0)).zero_()
                for idx in range(indicies.size(0)):   # pairing score with corresponding priorbox indices for negatives
                    score_tensor[idx] = all_max_scores[i][indices[idx]] # max score out of all classes for that corresponding index (relative to num_priors) # scores
                    index_tensor[idx] = indices[idx]  # indices relative to num_priors

                num_neg = math.min(math.floor((num_pos*self.neg_pos_ratio)+0.5),indices.size(0)) # round operation need to change this
                neg_i, ind = torch.topk(score_tensor,num_neg,1,True,True)
                neg_indices.resize_(num_neg)
                for number in range(num_neg):
                    neg_indices[number] = index_tensor[ind[number]]
                num_negs += num_neg
            self.all_match_indices.append(match_indices)  # store our matching indices for this image (relative to num_priors)
            self.all_neg_indices.append(neg_indices)  # store our negative indices for this image (relative to num_priors)


        loc_pred_data = torch.Tensor()
        loc_gt_data = torch.Tensor()
        if self.num_matches >= 1:
            # Form data to pass on to loc_loss_layer_.
            loc_pred_data = self.conf_data.new().resize_(self.num_matches*4).fill_(0)
            loc_gt_data = self.conf_data.new().resize_(self.num_matches*4).fill_(0)


            count = 0
            for i in range(num):
                loc_it = torch.Tensor(self.all_match_indices[i])
                for j in range(loc_it.size(0)): # skip over indices that were not matched
                    if loc_it[j] == -1:
                        continue

                    start_idx = j*4 # (was (j-1)*4 + 1)
                    # Store location prediction
                    assert(j <= num_priors)
                    loc_pred_data[count*4] = self.loc_data[i][start_idx]
                    loc_pred_data[count*4 + 1] = self.loc_data[i][start_idx+1]
                    loc_pred_data[count*4 + 2] = self.loc_data[i][start_idx+2]
                    loc_pred_data[count*4 + 3] = self.loc_data[i][start_idx+3]
                    # Store encoded ground truth

                    gt_idx = loc_it[j]  # ground truth index with respect to the number of ground truths for the given image

                    assert(gt_idx <= len(all_gt_bboxes[i])
                    gt_bbox = all_gt_bboxes[i][gt_idx] # The corresponding ground truth bbox

                    assert(j <= len(prior_bboxes))
                    gt_encode = encodeBbox(prior_bboxes[j], prior_variances[j], self.code_type,self.encode_variance_in_target, gt_bbox)
                    loc_gt_data[count*4] = gt_encode.minX
                    loc_gt_data[count*4 + 1] = gt_encode.minY
                    loc_gt_data[count*4 + 2] = gt_encode.maxX
                    loc_gt_data[count*4 + 3] = gt_encode.maxY

                    count += 1


      # Form data to pass on to conf_loss_layer.
        num_conf = 0
        if self.do_neg_mining:
            num_conf = self.num_matches + num_negs  # since our num_negs is 3x our num_matches, num_conf = 4 x num_matches
        else: # have not implemented this case
            num_conf = num * num_priors

        conf_pred_data = torch.Tensor()
        conf_gt_data = torch.Tensor()
        if num_conf >= 1:   # Reshape the confidence data.
            conf_gt_data = self.conf_data.new().resize_(num_conf).fill_(0)
            conf_pred_data = tself.conf_data.new().resize_(num_conf).fill_(0)

            count = 0
            for i in range(num):
                # Save matched (positive) bboxes scores and labels.
                conf_it = torch.Tensor(self.all_match_indices[i])  -- we are going to iterate through the matches stored for this image
                assert(conf_it.size(0) == num_priors)
                for j in range (num_priors):
                    if conf_it[j] == -1:  # skip over the indices that were not matched for now
                        continue

                    gt_label = all_gt_bboxes[i][conf_it[j]].label
                    local idx = count
                    conf_gt_data[idx] = gt_label
                    if self.do_neg_mining:  # Copy scores for matched bboxes.
                        conf_pred_data[count] = self.conf_data[i][j].clone()

                        count += 1

            if self.do_neg_mining:  # Save negative bbox scores and labels
                for n in range(self.all_neg_indices[i].size(0)):
                    j = self.all_neg_indices[i][n]
                    assert(j <= num_priors)
                    conf_pred_data[count] = self.conf_data[i][j].clone()
                    conf_gt_data[count] = self.background_label_id
                    count +=1



    #   self.softmax_layer = nn.LogSoftMax():cuda()
    #   self.softIn = conf_pred_data
    #   local scores = self.softmax_layer:forward(conf_pred_data)

      self.output.append(loc_pred_data)
      self.output.append(loc_gt_data)
      self.output.append(scores)
      self.output.append(conf_gt_data)

      return self.output
