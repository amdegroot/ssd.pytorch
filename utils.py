import torch
import torch.nn as nn
import math
if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn


def decode_boxes(prior_boxes,prior_variances,boxes,variance_encoded_in_target = False):
    """The function decodes predictions by restoring the learned offset
    in the bounding boxes.  At train time we perform the opposite encode
    function.

    Args:
        prior_boxes: (tensor) default boxes from priorbox layer, Shape: [num_priors,4].
        prior_variances: (tensor) variances corresponding to prior_box coords,
            Shape: [num_priors,4].
        boxes: (tensor) location prediction boxes from loc layers, Shape: [num_priors,4]
        variance_encoded_in_target: optional flag indicated if we encode variance in
            the ground truth's at train time. False means encoded in loc preds.
    """
    assert(prior_boxes.size(0) == prior_variances.size(0))
    assert(prior_boxes.size(0) == boxes.size(0))
    num_boxes = prior_boxes.size(0)
    decode = torch.Tensor(boxes.size(0), boxes.size(1))
    #decode = Variable(decode)
    if num_boxes >= 1:
        assert(prior_variances[0].size(0) == 4)

    prior_center_x = torch.add(prior_boxes[:,0],prior_boxes[:,2]).mul(0.5)
    prior_center_y = torch.add(prior_boxes[:,1],prior_boxes[:,3]).mul(0.5)
    p_w = prior_boxes[:,2] - prior_boxes[:,0]
    p_h = prior_boxes[:,3] - prior_boxes[:,1]

    if variance_encoded_in_target:
        # variance is encoded in target, we simply need to restore the offset predictions.
        torch.add(boxes[:,0], p_w, out=prior_center_x)
        torch.add(boxes[:,1], p_h, out=prior_center_y)
        decode_w = torch.addcmul(0.5,torch.exp(boxes[:,2]),p_w)
        decode_h = torch.addcmul(0.5,torch.exp(boxes[:,3]),p_h)

    else:
        # variance is encoded in bbox, we need to scale the offset accordingly.
        decode_center_x = boxes[:,0].mul(p_w).mul(prior_variances[0][0])
        decode_center_x += prior_center_x
        decode_center_y = boxes[:,1].mul(p_h).mul(prior_variances[0][1])
        decode_center_y += prior_center_y
        decode_w = torch.exp(boxes[:,2].mul(prior_variances[0][2]))*p_w.mul(0.5)
        decode_h = torch.exp(boxes[:,3].mul(prior_variances[0][3]))*p_h.mul(0.5)
    # instead just torch.cat() these or stack them to avoid Variable() issues
    decode[:,0] = prior_center_x - decode_w     # set xmin
    decode[:,1] = prior_center_y - decode_h     # set ymin
    decode[:,2] = prior_center_x + decode_w     # set xmax
    decode[:,3] = prior_center_y + decode_h  # set ymax
    return decode


def center_size(priors):
    """ Convert prior_boxes to (center_x, center_y, offset_x, offset_y)
    representation for comparison to loc data.

    Args:
        priors: (tensor) default boxes from priorbox layers.

    Return:
        priors: (tensor) Converted center,offset representation of priors.

    """
    return torch.cat((priors[:,:2] - priors[:,2:]/2,
                        priors[:,:2] + priors[:,2:]/2), 1)


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.

    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].

    """
    A = box_a.size(0)
    B = box_b.size(0)
    return torch.clamp(
        (torch.max(box_a[:,:2].unsqueeze(1).expand(A,B,2),
        box_b[:,:2].unsqueeze(0).expand(A,B,2))
        - torch.min(box_a[:,2:].unsqueeze(1).expand(A,B,2),
        box_b[:,2:].unsqueeze(0).expand(A,B,2))),
        min=0).prod(2).squeeze(2) # [A,B] multiplies width & height for each A,B


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.

    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)

    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]

    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:,2]-box_a[:,0])*(box_a[:,3]-box_a[:,1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:,2]-box_b[:,0])*(box_b[:,3]-box_b[:,1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union # [A,B]


def match(truths, priors, variances, labels, threshold):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices corresp.
    to both confidence and location preds.

    Args:
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        threshold: (float) The overlap threshold used when mathing boxes.

    Return:
        The matched indices corresponding to 1)locationand 2)confidence preds.

    """
    overlaps = jaccard(  # [num_obj,num_priors]
        truths,
        center_size(priors)
    )
    best_overlaps, best_idx = overlaps.max(0)  # [#num_priors,1]
    print(truths)
    print(best_idx)
    # matches = truths[best_idx.squeeze()]                 # [#num_priors,4]
    matches = truths[best_idx.squeeze().data]
    conf = labels[best_idx.squeeze().data].add(1)             # [num_priors,], bkg class = 0
    print(conf.size())
    print(overlaps.size())
    conf[best_overlaps.squeeze()<threshold] = 0               # label as background
    loc = encode(matches,priors,variances)     # [num_priors,4]

    # encoded location of each bounding box, label matched with each bounding box
    return loc, conf.long()


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.

    Args:
        matched: (tensor) Coordinates of ground truth for each prior,
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes after conversion to center,offset repr,
            Shape: [num_priors,4].

    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]

    """
    variances = [0.1,0.2]
    cx_cy = (matched[:,:2] + matched[:,2:]) / 2 - priors[:,:2] # dist b/t match center and prior's center
    cx_cy /= (variances[0] * priors[:,2:]) # encode variance
    wh = (matched[:,2:] - matched[:,:2]) / priors[:,2:] # match wh / prior wh
    wh = torch.log(wh) / variances[1]
    return torch.cat([cx_cy, wh], 1)  # [num_priors,4]


def apply_nms(boxes, scores, overlap, top_k):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.

    Args:
        boxes: (tensor) The location preds for the image, Shape: [num_priors,4].
        scores: (tensor) The class predicted scores for the image, Shape:[num_priors]
        overlap: (float) The overlap threshold for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.

    Return:
        The indices of the picked boxes with respect to num_priors.
    """

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


def sort(score_pairs, indices_list, label_list, ktk, final_scores, final_indices, final_labels):
    res, i = torch.topk(score_pairs,ktk,0,True,True)
    for n in range(ktk):
        final_scores[n] = res[n]
        final_indices[n] = indices_list[i[n]]
        final_labels[n] = label_list[i[n]]
