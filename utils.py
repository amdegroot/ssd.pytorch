import torch
import torch.nn as nn
import math
if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn

def decode_boxes(prior_bboxes,prior_variances,bboxes,variance_encoded_in_target = False):
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

    prior_center_x = torch.add(p_x1,p_x2).mul(0.5)
    prior_center_y = torch.add(p_y1,p_y2).mul(0.5)
    p_w-=p_x1
    p_h-=p_y1

    if variance_encoded_in_target:
        # variance is encoded in target, we simply need to restore the offset predictions.
        torch.add(b_x1, p_w, out=prior_center_x)
        torch.add(b_y1, p_h, out=prior_center_y)
        decode_w = torch.addcmul(0.5,torch.exp(b_x2),p_w)
        decode_h = torch.addcmul(0.5,torch.exp(b_y2),p_h)

    else:
        # variance is encoded in bbox, we need to scale the offset accordingly.
        decode_center_x = b_x1.mul(p_w).mul(var1)
        decode_center_x += prior_center_x
        decode_center_y = b_y1.mul(p_h).mul(var2)
        decode_center_y += prior_center_y
        decode_w = torch.exp(b_x2.mul(var3))*p_w.mul(0.5)
        decode_h = torch.exp(b_y2.mul(var4))*p_h.mul(0.5)

    decode[:,0] = prior_center_x - decode_w     # set xmin
    decode[:,1] = prior_center_y - decode_h     # set ymin
    decode[:,2] = prior_center_x + decode_w     # set xmax
    decode[:,3] = prior_center_y.add(decode_h)  # set ymax
    return decode

def center_size(priors):
     return torch.cat([priors[:,:2] - priors[:,2:]/2,
                       priors[:,:2] + priors[:,2:]/2], 1)

def intersect(box_a, box_b):
    '''
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection tensor Shape: [A,B].

    both tensors to [A,B,2]
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    subtracts max from min vals for each dimension from each tensor to get width and height,
    thresholds at zero, and returns a [A,B] tensor of width*height values
    '''
    A = box_a.size(0)
    B = box_b.size(0)

    return torch.clamp(
        box_a[:,:2].unsqueeze(1).expand(A,B,2).max(
        box_b[:,:2].unsqueeze(0).expand(A,B,2),)
        - box_a[:,2:].unsqueeze(1).expand(N,M,2).min(
        box_b[:,2:].unsqueeze(0).expand(N,M,2)),
        min=0).prod(2) # [A,B] multiplies width & height for each A,B


def jaccard(box_a, box_b):
    inter = intersect(boxa_a, box_b)
    area_a = (box_a[:,2]-box_a[:,0])*(box_a[:,3]-box_a[:,1]).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = (box_b[:,2]-box_b[:,0])*(box_b[:,3]-box_b[:,1]).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union # [A,B]


def match(truths, priors, variances, labels, threshold):
    overlaps = jaccard(  # [#obj,8732]
        truths,
        center_size(priors)
    )
    best_overlaps, best_idx = overlaps.max(0)  # [#8732,1]
    best_idx.squeeze_(0)                       # [#8732]
    best_overlaps.squeeze_(0)             # [#8732]
    matches = truths[best_idx]                 # [#8732,4]

    conf = classes[max_idx].add(1)   # [8732,], background class = 0
    conf[overlaps<threshold] = 0       # background
    loc = encode(matches,priors,variances) # [8732, 4]

    return loc, conf # encoded location of each bounding box, label matched with each bounding box


def encode(matched, priors, variances):
    # matched [8732,4] (x1,y1,x2,y2) ... coords of ground truth for each prior
    # priors  [8732,4] (cx,cy,w,h)
    # encoding variance in bounding boxes
    cx_cy = (matched[:,:2] + matched[:,2:]) / 2 - priors[:,:2] # dist b/t match center and prior's center
    cx_cy /= (variances[0] * priors[:,2:]) # encode variance
    wh = (matched[:,2:] - matched[:,:2]) / priors[:,2:] # match wh / prior wh
    wh = torch.log(wh) / variances[1]
    return torch.cat([cx_cy, wh], 1)  # [8732,4]

def apply_nms(boxes, scores, overlap, top_k):


    pick = torch.zeros(scores.size(0))
    if boxes.numel() == 0:
        return pick
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = torch.mul(x2 - x1 + 1, y2 - y1 + 1)
    v, I = scores.sort(0) # sort in ascending order
    I = I[-top_k:]

    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while I.numel() > 0:
        last = I.size(0)
        i = I[last-1]

        pick[count] = i
        count += 1
        if last == 1:
            break

        I = I[:-1] # remove picked element from view

        # load values
        torch.index_select(x1, 0, I, out=xx1)
        torch.index_select(y1, 0, I, out=yy1)
        torch.index_select(x2, 0, I, out=xx2)
        torch.index_select(y2, 0, I, out=yy2)

        # TODO: time comparison using map_() and xx1 < x1[i] instead
        # store element-wise max with next highest score
        torch.clamp(xx1,min = x1[i])
        torch.clamp(yy1,min = y1[i])
        torch.clamp(xx2,max = x2[i])
        torch.clamp(yy2,max = y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)

        w = xx2 - xx1
        h = yy2 - yy1
        w += 1
        h +=1
        w.max(torch.zeros(w.size(0)))
        h.max(torch.zeros(h.size(0)))

        # reuse existing tensors
        inter = w*h
        # IoU .= i / (area(a) + area(b) - i)
        xx1 = torch.index_select(area, 0, I) # load remaining areas into xx1
        IoU = inter.div(xx1 + area[i] - inter) # store result in iou
        mask = IoU.le(overlap)

        # keep only elements with a IoU < overlap
        I = torch.masked_select(I,IoU.le(overlap))

    # reduce size to actual count
    return pick[:count]


def hard_negative_mining(self, conf_loss, pos):
        '''Return negative indices that is 3x the number as postive indices.
        Args:
          conf_loss: (tensor) cross entroy loss between conf_preds and conf_targets, sized [N*8732,].
          pos: (tensor) positive(matched) box indices, sized [N,8732].
        Return:
          (tensor) negative indices, sized [N,8732].
        '''
        batch_size, num_boxes = pos.size()

        conf_loss[pos] = 0  # set pos boxes = 0, the rest are neg conf_loss
        conf_loss = conf_loss.view(batch_size, -1)  # [N,8732]
        max_loss,_ = conf_loss.sort(1, descending=True)  # sort by neg conf_loss

        num_pos = pos.long().sum(1)  # [N,1]
        num_neg = torch.clamp(3*num_pos, max=num_boxes-1)  # [N,1]

        pivot_loss = max_loss.gather(1, num_neg)           # [N,1]
        neg = conf_loss > pivot_loss.expand_as(conf_loss)  # [N,8732]
        return neg


def cross_entropy_loss(self, x, y):
        '''Cross entropy loss w/o averaging across all samples.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) cross entroy loss, sized [N,].
        '''
        xmax = x.data.max()
        log_sum_exp = torch.log(torch.sum(torch.exp(x-xmax), 1)) + xmax
        return log_sum_exp - x.gather(1, y.view(-1,1))


def sort(score_pairs, indices_list, label_list, ktk, final_scores, final_indices, final_labels):
    res, i = torch.topk(score_pairs,ktk,0,True,True) 
    for n in range(ktk):
        final_scores[n] = res[n]
        final_indices[n] = indices_list[i[n]]
        final_labels[n] = label_list[i[n]]
