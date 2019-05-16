# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""

import numpy as np
from utils.cython_bbox import bbox_overlaps
from detection_opr.utils.bbox_transform import bbox_transform
from config import cfg
from IPython import embed

# todo make anchors maxsize not check
def make_anchors_func():
    sizes = cfg.sizes
    ratios = cfg.ratios
    anchor_layers = []

    for k, fm_shape in enumerate(cfg.feat_shapes):
        size = sizes[k]
        ratio = ratios[k]
        min_size = size[0]
        max_size = size[1]

        if max_size:
            num_anchors = len(ratio) + 2
        else:
            num_anchors = len(ratio) + 1

        layer_height = cfg.feat_shapes[k][0]
        layer_width = cfg.feat_shapes[k][1]

        step_x = 1. / layer_width
        step_y = 1. / layer_height

        prior_boxes = np.empty(
            (layer_width * layer_height * num_anchors, 4), dtype=np.float32)

        count = 0
        for r in range(layer_height):
            center_y = (r + 0.5) * step_y
            for c in range(layer_width):
                center_x = (c + 0.5) * step_x

                w = min_size / 2.
                h = min_size / 2.
                prior_boxes[count] = np.array([
                    center_x - w, center_y - h, center_x + w, center_y + h])
                count += 1

                if max_size:
                    w = max_size / 2.
                    h = max_size / 2.
                    prior_boxes[count] = np.array([
                        center_x - w, center_y - h, center_x + w, center_y + h])
                    count += 1

                for ar in ratio:
                    w = min_size * np.sqrt(ar) / 2.
                    h = min_size / np.sqrt(ar) / 2.
                    prior_boxes[count] = np.array([
                        center_x - w, center_y - h, center_x + w, center_y + h])
                    count += 1
        prior_boxes = prior_boxes.clip(0., 1.)
        anchor_layers.append(prior_boxes)

    return np.concatenate(anchor_layers, axis=0)
