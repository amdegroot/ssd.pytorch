"""
given a pascal voc imdb, compute mAP

MIT License

Copyright (c) 2016 Joshua Z. Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

https://github.com/zhreshold/mxnet-ssd/blob/67ba1c9a8cb0508b018e958fd540f1e99423d0e3/evaluate/eval_voc.py
"""
from __future__ import print_function
import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle


def parse_voc_rec(filename):
    """
    parse pascal voc record into a dictionary
    :param filename: xml file path
    :return: list of dict
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_dict = dict()
        obj_dict['name'] = obj.find('name').text
        obj_dict['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [int(bbox.find('xmin').text),
                            int(bbox.find('ymin').text),
                            int(bbox.find('xmax').text),
                            int(bbox.find('ymax').text)]
        objects.append(obj_dict)
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imageset_file, classname, cache_dir, ovthresh=0.5, use_07_metric=False):
    """
    pascal voc evaluation
    :param detpath: detection results detpath.format(classname)
    :param annopath: annotations annopath.format(classname)
    :param imageset_file: text file containing list of images
    :param classname: category name
    :param cache_dir: caching annotations
    :param ovthresh: overlap threshold
    :param use_07_metric: whether to use voc07's 11 point ap computation
    :return: rec, prec, ap
    """
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    cache_file = os.path.join(cache_dir, 'annotations.pkl')
    with open(imageset_file, 'r') as f:
        lines = f.readlines()
    image_filenames = [x.strip() for x in lines]

    # load annotations from cache
    if not os.path.isfile(cache_file):
        recs = {}
        for ind, image_filename in enumerate(image_filenames):
            recs[image_filename] = parse_voc_rec(annopath.format(image_filename))
            if ind % 100 == 0:
                print('reading annotations for {:d}/{:d}'.format(ind + 1, len(image_filenames)))
        print('saving annotations cache to {:s}'.format(cache_file))
        with open(cache_file, 'w') as f:
            pickle.dump(recs, f)
    else:
        with open(cache_file, 'r') as f:
            recs = pickle.load(f)

    # extract objects in :param classname:
    class_recs = {}
    npos = 0
    for image_filename in image_filenames:
        objects = [obj for obj in recs[image_filename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in objects])
        difficult = np.array([x['difficult'] for x in objects]).astype(np.bool)
        det = [False] * len(objects)  # stand for detected
        npos = npos + sum(~difficult)
        class_recs[image_filename] = {'bbox': bbox,
                                      'difficult': difficult,
                                      'det': det}

    # read detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    bbox = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_inds = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    bbox = bbox[sorted_inds, :]
    image_ids = [image_ids[x] for x in sorted_inds]

    # go down detections and mark true positives and false positives
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        r = class_recs[image_ids[d]]
        bb = bbox[d, :].astype(float)
        ovmax = -np.inf
        bbgt = r['bbox'].astype(float)

        if bbgt.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(bbgt[:, 0], bb[0])
            iymin = np.maximum(bbgt[:, 1], bb[1])
            ixmax = np.minimum(bbgt[:, 2], bb[2])
            iymax = np.minimum(bbgt[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                   (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not r['difficult'][jmax]:
                if not r['det'][jmax]:
                    tp[d] = 1.
                    r['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid division by zero in case first detection matches a difficult ground ruth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
