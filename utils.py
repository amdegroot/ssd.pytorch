import torch
import torch.nn as nn
import math
def decode_boxes(prior_bboxes,prior_variances,code_type,variance_encoded_in_target,bboxes):
    bboxes = bboxes.view(-1,4)
    assert(prior_bboxes.size(0) == prior_variances.size(0))

    assert(prior_bboxes.size(0) == bboxes.size(0))
    num_bboxes = prior_bboxes.size(0)
    decode = torch.Tensor(bboxes.size(0), bboxes.size(1))
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
        decode_center_x = torch.Tensor()
        decode_center_y = torch.Tensor()
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
            torch.addcmul(prior_center_x,var1,b_x1,p_w)
            torch.addcmul(prior_center_y,var2,b_y1,p_h)
            decode_w = torch.exp(b_x2.mul(var3))*p_w.mul(0.5)
            decode_h = torch.exp(b_y2.mul(var4))*p_h.mul(0.5)

        decode[:,0] = prior_center_x.clone()-decode_w# set xmin
        decode[:,1] = prior_center_y.clone()-decode_h# set ymin
        decode[:,2] = prior_center_x.add(decode_w)# set xmax
        decode[:,3] = prior_center_y.add(decode_h)# set ymax
    else:
       print('<Detection_Output> Unknown LocLossType')
    return decode


def apply_nms(boxes, scores, overlap, top_k):
    pick = torch.LongTensor()
    if boxes.numel() == 0:
        return pick

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    #print(scores.size())
    area = torch.mul(x2 - x1 + 1, y2 - y1 + 1)

    v, I = torch.sort(scores) # sort in ascending order

    I = I[I.size(0)-top_k:I.size(0)] # only want top k


    pick.resize_(v.size(0)).zero_()
    count = 0

    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()

    w = boxes.new()
    h = boxes.new()

    while I.numel() > 0:
        last = I.size(0)
        i = I[last-1]

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

    w.resize_(xx2.size())
    h.resize_(yy2.size())
    torch.add(xx2, -1, xx1, out = w).add(1).max(0)
    torch.add(yy2, -1, yy1, out = h).add(1).max(0)

    # reuse existing tensors
    inter = torch.mul(w,h)
    IoU = h

    # IoU .= i / (area(a) + area(b) - i)
    xx1 = torch.index_select(area, 0, I) # load remaining areas into xx1
    IoU = inter / (xx1 + area[i] - inter) # store result in iou

    I = I[IoU.le(overlap)]
    # keep only elements with a IoU < overlap

    # reduce size to actual count
    pick = pick[0:count-1]
    return pick

def sort(score_pairs, indices_list, label_list, ktk, final_scores, final_indices, final_labels):
       # note. removed tk field which was -1 for first case and length for second
    res, i = torch.topk(score_pairs,ktk,0,True,True) # maybe change 0 back to 1 (changed for python)
    for n in range(ktk):
        final_scores[n] = res[n]
        final_indices[n] = indices_list[i[n]]
        final_labels[n] = label_list[i[n]]
