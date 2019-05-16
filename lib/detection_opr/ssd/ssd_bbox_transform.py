# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np


def bbox_transform(anchors, gt):
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5

    gw = gt[:, 2] - gt[:, 0]
    gh = gt[:, 3] - gt[:, 1]
    gx = (gt[:, 0] + gt[:, 2]) * 0.5
    gy = (gt[:, 1] + gt[:, 3]) * 0.5

    targets_dx = (gx - ax) / aw / 0.1
    targets_dy = (gy - ay) / ah / 0.1
    targets_dw = np.log(gw / aw) / 0.2
    targets_dh = np.log(gh / ah) / 0.2

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    targets[np.where(np.isnan(targets))] = 0
    return targets


def bbox_transform_inv(boxes, deltas):
    aw = boxes[:, 2] - boxes[:, 0]
    ah = boxes[:, 3] - boxes[:, 1]
    ax = (boxes[:, 0] + boxes[:, 2]) / 2.
    ay = (boxes[:, 1] + boxes[:, 3]) / 2.

    ox = deltas[:, 0] * aw * 0.1 + ax
    oy = deltas[:, 1] * ah * 0.1 + ay
    ow = np.exp(deltas[:, 2] * 0.2) * aw / 2.
    oh = np.exp(deltas[:, 3] * 0.2) * ah / 2.

    targets = np.vstack((ox - ow, oy - oh, ox + ow, oy + oh)).transpose()

    targets[np.where(np.isnan(targets))] = 0

    return targets


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    (height, width)
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
