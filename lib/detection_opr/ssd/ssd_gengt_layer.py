# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""

import numpy as np
from utils.cython_bbox import bbox_overlaps, bbox_overlaps_float
from detection_opr.ssd.ssd_bbox_transform import bbox_transform
from config import cfg
from IPython import embed


def ssd_gengt_layer(batch_pred_conf, prior_boxes, batch_gt_boxes):
    batch_labels = []
    batch_deltas = []
    overlap_threshold = cfg.TRAIN.overlap_threshold
    negative_mining_thresh = cfg.TRAIN.neg_overlap

    for tl in range(len(batch_gt_boxes)):
        pred_conf = batch_pred_conf[tl]
        gt_boxes = batch_gt_boxes[tl]
        first_ignore = np.argmax(np.fabs(gt_boxes[:, 0] - -1) < 1e-3)
        if np.fabs(gt_boxes[first_ignore, 0] - -1) < 1e-3:
            gt_boxes = gt_boxes[:first_ignore]

        num_gt_boxes = len(gt_boxes)
        num_anchors = len(prior_boxes)
        num_positive = 0

        overlaps = bbox_overlaps(
            np.ascontiguousarray(prior_boxes * cfg.image_size, dtype=np.float),
            np.ascontiguousarray(gt_boxes * cfg.image_size, dtype=np.float))
        # overlaps = bbox_overlaps_float(
        #    np.ascontiguousarray(prior_boxes, dtype=np.float),
        #    np.ascontiguousarray(gt_boxes, dtype=np.float))

        anchor_flags = np.empty((len(prior_boxes),), dtype=np.int32)
        anchor_flags.fill(-1)
        gt_flags = np.empty((len(prior_boxes),), dtype=np.bool)
        gt_flags.fill(False)

        max_matches_iou = np.empty((len(prior_boxes),), dtype=np.float32)
        max_matches_iou.fill(-1.0)
        max_matches_gtid = np.empty((len(prior_boxes),), dtype=np.int32)
        max_matches_gtid.fill(-1)

        # gt_boxes match priors
        queues = []
        queue_tops = []
        for i in range(len(gt_boxes)):
            inds = np.argpartition(
                overlaps[:, i], num_anchors - num_gt_boxes)[-num_gt_boxes:]
            sort_inds = np.argsort(overlaps[inds, i])[::-1]
            queues.append(inds[sort_inds])
            queue_tops.append(0)

        for i in range(num_gt_boxes):
            max_overlap = 1e-6
            best_gt = -1
            best_anchor = -1

            for j in range(num_gt_boxes):
                if gt_flags[j]:
                    continue
                while anchor_flags[queues[j][queue_tops[j]]] != -1:
                    queue_tops[j] += 1

                _anchor = queues[j][queue_tops[j]]
                if max_overlap < overlaps[_anchor][j]:
                    max_overlap = overlaps[_anchor][j]
                    best_gt = j
                    best_anchor = _anchor

            anchor_flags[best_anchor] = 1
            gt_flags[best_gt] = True
            max_matches_iou[best_anchor] = max_overlap
            max_matches_gtid[best_anchor] = best_gt
            num_positive += 1

        anchor_argmax_iou = overlaps.argmax(axis=1)
        anchor_max_iou = overlaps[np.arange(num_anchors), anchor_argmax_iou]
        # priors match gt_boxes
        if overlap_threshold > 0:
            inds = np.where((anchor_max_iou > 1e-6) & (anchor_flags != 1))
            max_matches_iou[inds] = anchor_max_iou[inds]
            max_matches_gtid[inds] = anchor_argmax_iou[inds]

            inds = np.where(
                (anchor_max_iou > overlap_threshold) & (anchor_flags != 1))
            gt_flags[anchor_argmax_iou[inds]] = True
            anchor_flags[inds] = 1
            num_positive += len(inds[0])

        # Negative mining
        max_pred_conf_head = np.max(pred_conf, axis=1, keepdims=True)
        pred_conf = np.exp(pred_conf - max_pred_conf_head)
        max_pred_conf = np.max(
            pred_conf[:, 1:], axis=1, keepdims=True) / \
                        np.sum(pred_conf, axis=1, keepdims=True)

        if cfg.TRAIN.do_neg_mining:
            num_negative = int(num_positive * cfg.TRAIN.neg_pos_ratio)
            if num_negative > (num_anchors - num_positive):
                num_negative = num_anchors - num_positive
            if num_negative > 0:
                inds = np.where((anchor_flags != 1) & (
                    max_matches_iou < negative_mining_thresh))

                max_matches_iou[inds] = anchor_max_iou[inds]
                max_matches_gtid[inds] = anchor_argmax_iou[inds]

                neg_inds = np.where((anchor_flags != 1) & (
                    max_matches_iou < negative_mining_thresh))[0]

                order = max_pred_conf[neg_inds].argsort(axis=0)[::-1]
                anchor_flags[neg_inds[order[:num_negative, 0]]] = 0

        labels = np.array(anchor_flags)
        inds = np.where(anchor_flags == 1)
        labels[inds] = gt_boxes[max_matches_gtid[inds], 4]

        deltas = bbox_transform(
            prior_boxes,
            gt_boxes[max_matches_gtid, :][:, :4].astype(
                np.float32, copy=False))

        batch_labels.append(labels)
        batch_deltas.append(deltas)

    return np.asarray(batch_labels, dtype=np.int32), np.asarray(
        batch_deltas, dtype=np.float32)
