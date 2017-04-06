import torch
import numpy as np
from . import meter


class VOC07APMeter(meter.Meter):
    def __init__(self, ovp_thresh=0.5, use_difficult=False,
                 class_names=None, pred_idx=0):
        self.use_difficult = use_difficult
        self.class_names = class_names
        self.pred_idx = int(pred_idx)
        self.reset()

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.pred_boxes = torch.LongTensor(torch.LongStorage())
        self.target_boxes = torch.LongTensor(torch.LongStorage())
        self.true_positives = torch.FloatTensor(torch.FloatStorage())
        self.false_positives = torch.FloatTensor(torch.FloatStorage())

        self.n_gtboxes = 0
        self.d_n_gbboxes = {}
        self.d_tp = {}
        self.d_fp = {}

    def add(self, labels, preds):
        # independant execution for each image
        for i in range(labels[0].shape[0]):
            # get as numpy arrays
            label = labels[0][i].asnumpy()
            pred = preds[self.pred_idx][i].asnumpy()
            # calculate for each class
            while (pred.shape[0] > 0):
                cid = int(pred[0, 0])
                indices = np.where(pred[:, 0].astype(int) == cid)[0]
                if cid < 0:
                    pred = np.delete(pred, indices, axis=0)
                    continue
                dets = pred[indices]
                pred = np.delete(pred, indices, axis=0)
                # sort by score, desceding
                dets[dets[:, 1].argsort()[::-1]]
                records = np.hstack((dets[:, 1][:, np.newaxis],
                                     np.zeros((dets.shape[0], 1))))
                # ground-truths
                gts = label[np.where(label[:, 0].astype(int) == cid)[0], :]
                if gts.size > 0:
                    found = [False] * gts.shape[0]
                    for j in range(dets.shape[0]):
                        # compute overlaps
                        ious = iou(dets[j, 2:], gts[:, 1:5])
                        ovargmax = np.argmax(ious)
                        ovmax = ious[ovargmax]
                        if ovmax > self.ovp_thresh:
                            if (not self.use_difficult and
                                gts.shape[1] >= 6 and
                                gts[ovargmax, 5] > 0):
                                pass
                            else:
                                if not found[ovargmax]:
                                    records[j, -1] = 1  # tp
                                    found[ovargmax] = True
                                else:
                                    # duplicate
                                    records[j, -1] = 2  # fp
                        else:
                            records[j, -1] = 2 # fp
                else:
                    # no gt, mark all fp
                    records[:, -1] = 2

                # ground truth count
                if (not self.use_difficult and gts.shape[1] >= 6):
                    gt_count = np.sum(gts[:, 5] < 1)
                else:
                    gt_count = gts.shape[0]

                # now we push records to buffer
                # first column: score, second column: tp/fp
                # 0: not set(matched to difficult or something), 1: tp, 2: fp
                records = records[np.where(records[:, -1] > 0)[0], :]
                if records.size > 0:

                    self._insert(cid, records, gt_count)

    def value(self):
        """Get the current evaluation result.
        Returns
        -------
        name : str
        Name of the metric.
        value : float
        Value of the evaluation.
        """
        aps = []
        for k, v in self.records.items():
            recall, prec = self._recall_prec(v, self.counts[k])
            ap = self._average_precision(recall, prec)
            aps.append(ap)
            if self.num is not None and k < (self.num - 1):
                self.sum_metric[k] = ap
                self.num_inst[k] = 1
            if self.num is None:
                self.num_inst = 1
                self.sum_metric = np.mean(aps)
            else:
                self.num_inst[-1] = 1
                self.sum_metric[-1] = np.mean(aps)
        if self.num is None:
            if self.num_inst == 0:
                return (self.class_names, float('nan'))
            else:
                return (self.class_names, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.class_names[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)

    def _average_precision(self, rec, prec):
        """
        calculate average precision, override the default one,
        special 11-point metric
        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
        return ap

    def _insert(self, key, records, count):
        """ Insert records according to key """
        if key not in self.records:
            assert key not in self.counts
            self.records[key] = records
            self.counts[key] = count
        else:
            self.records[key] = np.vstack((self.records[key], records))
            assert key in self.counts
            self.counts[key] += count

    def _recall_prec(self, record, count):
        """ get recall and precision from internal records """
        sorted_records = record[record[:,0].argsort()[::-1]]
        tp = np.cumsum(sorted_records[:, 1].astype(int) == 1)
        fp = np.cumsum(sorted_records[:, 1].astype(int) == 2)
        if count <= 0:
            recall = tp * 0.0
        else:
            recall = tp / float(count)
        prec = tp.astype(float) / (tp + fp)
        return recall, prec
