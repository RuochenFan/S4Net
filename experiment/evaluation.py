# Authors: 644142239@qq.com (Ruochen Fan)
# Following https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

import pickle as pkl
import numpy as np
import cv2
from IPython import embed
from sklearn import metrics
from numba import jit

filter_thresh = 0.5

# @jit
def calc_iou(mask_a, mask_b):
    intersection = (mask_a+mask_b>=2).astype(np.float32).sum()
    iou = intersection / (mask_a+mask_b>=1).astype(np.float32).sum()
    return iou

# @jit
def calc_accu_recall(gt_masks, segmaps, iou_thresh, ious):
    num_TP = 0
    for i in range(len(gt_masks)):
        max_match = 0
        max_match_id = 0
        for j in range(len(segmaps)):
            if ious[i,j] > max_match:
                max_match = ious[i,j]
                max_match_id = j
        if max_match > iou_thresh:
            num_TP += 1
            ious[:, max_match_id] = 0

    recall = num_TP / len(gt_masks)
    accu = num_TP / len(segmaps)

    return recall, accu

def eval(reses, iou_thresh):
    aps = []
    aps_overlapped = []
    aps_divided = []

    for ind, res in enumerate(reses):
        gt_masks = res['gt_masks']
        segmaps = res['segmaps']    #ndarray，[20, w, h]
        scores = res['scores']      #ndarray，[20]

        order = np.argsort(scores)[::-1]
        scores = scores[order]
        segmaps = segmaps[order]
        segmaps = (segmaps > filter_thresh).astype(np.int32)

        ious = np.zeros([100, 100])
        for i in range(len(gt_masks)):
            for j in range(len(segmaps)):
                ious[i, j] = calc_iou(gt_masks[i], segmaps[j])

        recall_accu = {}
        for i in range(len(scores)):
            accu, recall = calc_accu_recall(gt_masks, segmaps[:i+1], iou_thresh, ious.copy())
            if recall in recall_accu:
                if accu > recall_accu[recall]:
                    recall_accu[recall] = accu
            else:
                recall_accu[recall] = accu
        
        recalls = list(recall_accu.keys())
        recalls.sort()
        accus = []
        for recall in recalls:
            accus.append(recall_accu[recall])
        accus = accus[:1] + accus
        recalls = [0] + recalls

        # for i in range(len(accus)):
        #     accus[i] = max(accus[i:])

        ap = metrics.auc(recalls, accus)
        aps.append(ap)

    return sum(aps) / len(aps)
