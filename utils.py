import torch
import torch.nn as nn
import math
#from torch.autograd import Variable
if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn

def decode_boxes(prior_bboxes,prior_variances,code_type,variance_encoded_in_target,bboxes):
    bboxes = bboxes.view(-1,4)
    assert(prior_bboxes.size(0) == prior_variances.size(0))

    assert(prior_bboxes.size(0) == bboxes.size(0))
    num_bboxes = prior_bboxes.size(0)
    decode = torch.Tensor(bboxes.size(0), bboxes.size(1))
    #decode = Variable(decode)
    if num_bboxes >= 1:
        assert(prior_variances[0].size(0) == 4)

    p_x1 = prior_bboxes[:,0]
    p_y1 = prior_bboxes[:,1]
    p_x2 = prior_bboxes[:,2]
    p_y2 = prior_bboxes[:,3]

    b_x1 = bboxes[:,0]
    b_y1 = bboxes[:,1]
    b_x2 = bboxes[:,2]
    b_y2 = bboxes[:,3]

    var1 = prior_variances[0][0]
    var2 = prior_variances[0][1]
    var3 = prior_variances[0][2]
    var4 = prior_variances[0][3]

    p_w = p_x2.clone()
    p_h = p_y2.clone()

    decode_w = p_w.new()
    decode_h = p_h.new()

    if code_type == 'CORNER':
        if variance_encoded_in_target:
            # variance is encoded in target, we simply need to add the offset predictions
            torch.add(decode[:,0], torch.add(p_x1, var1))
            torch.add(decode[:,1], torch.add(p_y1, var2))
            torch.add(decode[:,2], torch.add(p_x2, var3))
            torch.add(decode[:,3], torch.add(p_y2, var4))
        else:
            # variance is encoded in bbox, we need to scale the offset accordingly.
            torch.add(p_x1, var1, b_x1, out=decode[:,0])
            torch.add(p_y1, var2, b_y1, out=decode[:,1])
            torch.add(p_x2, var3, b_x2, out=decode[:,2])
            torch.add(p_y2, var4, b_y2, out=decode[:,3])

    elif code_type == 'CENTER':
        print('yo')
        #decode_center_x = torch.cuda.FloatTensor()
        #decode_center_y = torch.cuda.FloatTensor()
        prior_center_x = torch.add(p_x1,p_x2).mul(0.5)
        prior_center_y = torch.add(p_y1,p_y2).mul(0.5)
        p_w-=p_x1
        p_h-=p_y1


        if variance_encoded_in_target:
            # variance is encoded in target, we simply need to restore the offset predictions.
            print('its encoded')
            torch.add(b_x1, p_w, out=prior_center_x)
            torch.add(b_y1, p_h, out=prior_center_y)
            decode_w = torch.addcmul(0.5,torch.exp(b_x2),p_w)
            decode_h = torch.addcmul(0.5,torch.exp(b_y2),p_h)

        else:
            # variance is encoded in bbox, we need to scale the offset accordingly.
            # print(type(decode_center_x))

            # print(type(b_x1.data))
            # print(b_x1.data.size())
            # print(type(p_w.data))
            # print(p_w.data.size())
            # print(type(var1.data))
            # print(var1.data.size())
            # print(var1.data)
            # print(var1)

            decode_center_x = b_x1.mul(p_w).mul(var1)
            decode_center_x += prior_center_x
            decode_center_y = b_y1.mul(p_h).mul(var2)
            decode_center_y += prior_center_y
            # torch.addcmul(prior_center_x,var1,b_x1,p_w)
            # torch.addcmul(prior_center_y,var2,b_y1,p_h)
            decode_w = torch.exp(b_x2.mul(var3))*p_w.mul(0.5)
            decode_h = torch.exp(b_y2.mul(var4))*p_h.mul(0.5)

        decode[:,0] = prior_center_x-decode_w# set xmin
        decode[:,1] = prior_center_y-decode_h# set ymin
        decode[:,2] = prior_center_x.add(decode_w)# set xmax
        decode[:,3] = prior_center_y.add(decode_h)# set ymax
    else:
       print('<Detection_Output> Unknown LocLossType')
    return decode

def encode_bbox(prior_bbox, prior_variance, encode_variance_in_target, return_iou, bbox):
    #iou = self.iou(box)
    encoded_box = torch.zeros(self.num_priors, 4)
    #assign_mask = iou > overlap_threshold
    #if not assign_mask.any():
    #    assign_mask[iou.argmax()] = True
    #if return_iou:
    #    encoded_box[:, -1][assign_mask] = iou[assign_mask]
    #assigned_priors = self.priors[assign_mask]
    box_center = 0.5 * (box[:2] + box[2:])
    box_wh = box[2:] - box[:2]
    priors_center = 0.5 * (prior_bbox[:, :2] + prior_bbox[:, 2:4])
    priors_wh = (prior_bbox[:, 2:4] - prior_bbox[:, :2])
    # here we encode variance
    encoded_box[:, :2] = box_center - priors_center
    encoded_box[:, :2] /= priors_wh
    encoded_box[:, :2] /= assigned_priors[:, -4:-2]
    encoded_box[:, 2:4] = np.log(box_wh /priors_wh)
    encoded_box[:, 2:4] /= priors[:, -2:]
    return encoded_box.view(-1)

def iou(box):
    """Compute intersection over union for the box with all priors.
    # Arguments
        box: Box, numpy tensor of shape (4,).
    # Return
        iou: Intersection over union,
            numpy tensor of shape (num_priors).
    """
    # compute intersection
    inter_upleft = np.maximum(self.priors[:, :2], box[:2])
    inter_botright = np.minimum(self.priors[:, 2:4], box[2:])
    inter_wh = inter_botright - inter_upleft
    inter_wh = np.maximum(inter_wh, 0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    # compute union
    area_pred = (box[2] - box[0]) * (box[3] - box[1])
    area_gt = (self.priors[:, 2] - self.priors[:, 0])
    area_gt *= (self.priors[:, 3] - self.priors[:, 1])
    union = area_pred + area_gt - inter
    # compute iou
    iou = inter / union
    return iou


# def match_bbox()


def encode_box(box, return_iou=True):
    """Encode box for training, do it only for assigned priors.
    # Arguments
        box: Box, numpy tensor of shape (4,).
        return_iou: Whether to concat iou to encoded values.
    # Return
        encoded_box: Tensor with encoded box
            numpy tensor of shape (num_priors, 4 + int(return_iou)).
    """
    iou = iou(box)
    encoded_box = np.zeros((self.num_priors, 4 + return_iou))
    assign_mask = iou > self.overlap_threshold
    if not assign_mask.any():
        assign_mask[iou.argmax()] = True
    if return_iou:
        encoded_box[:, -1][assign_mask] = iou[assign_mask]
    assigned_priors = self.priors[assign_mask]
    box_center = 0.5 * (box[:2] + box[2:])
    box_wh = box[2:] - box[:2]
    assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                    assigned_priors[:, 2:4])
    assigned_priors_wh = (assigned_priors[:, 2:4] -
                          assigned_priors[:, :2])
    # we encode variance
    encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
    encoded_box[:, :2][assign_mask] /= assigned_priors_wh
    encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2]
    encoded_box[:, 2:4][assign_mask] = np.log(box_wh /
                                              assigned_priors_wh)
    encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:]
    return encoded_box.ravel()

def apply_nms(boxes, scores, overlap, top_k):
    pick = torch.Tensor()
    if boxes.numel() == 0:
        return pick

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    #print(scores.size())
    area = torch.mul(x2 - x1 + 1, y2 - y1 + 1)

    v, I = scores.sort(0) # sort in ascending order

    I = I[(I.size(0)-top_k):I.size(0)] # only want top k


    pick.resize_(v.size(0)).zero_()
    count = 0

    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()

    w = boxes.new()
    h = boxes.new()

    while I.numel() > 0:
        last = I.size(0)-1
        i = I[last]

        pick[count] = i
        count += 1

        if last == 1:
            break

        I = I[0:last-1] # remove picked element from view

    # load values
    xx1 = torch.index_select(x1, 0, I)
    yy1 = torch.index_select(y1, 0, I)
    xx2 = torch.index_select(x2, 0, I)
    yy2 = torch.index_select(y2, 0, I)

    # compute intersection area
    xx1 = torch.max(xx1,xx1.clone().fill_(x1[i]))
    yy1 = torch.max(yy1,yy1.clone().fill_(y1[i]))
    xx2 = torch.min(xx2,xx2.clone().fill_(x2[i]))
    yy2 = torch.min(yy2,yy2.clone().fill_(y2[i]))

    w.resize_as_(xx2)
    h.resize_as_(yy2)
    torch.add(xx2, -1, xx1, out = w).add(1).max(0)
    torch.add(yy2, -1, yy1, out = h).add(1).max(0)

    # reuse existing tensors
    inter = w*h
    IoU = h

    # IoU .= i / (area(a) + area(b) - i)
    xx1 = torch.index_select(area, 0, I) # load remaining areas into xx1
    IoU = inter.div(xx1 + area[i] - inter) # store result in iou

    I = I[IoU.le(overlap)]
    # keep only elements with a IoU < overlap

    # reduce size to actual count
    pick = pick[0:count-1]
    print(pick)
    return pick



def sort(score_pairs, indices_list, label_list, ktk, final_scores, final_indices, final_labels):
       # note. removed tk field which was -1 for first case and length for second
    res, i = torch.topk(score_pairs,ktk,0,True,True) # maybe change 0 back to 1 (changed for python)
    for n in range(ktk):
        final_scores[n] = res[n]
        final_indices[n] = indices_list[i[n]]
        final_labels[n] = label_list[i[n]]
