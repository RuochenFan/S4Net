# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""

from config import cfg
import numpy as np

from dataset_mx import *


def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        if 'max_overlaps' not in entry.keys():
            from IPython import embed;
            embed()
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print('Filtered {} roidb entries: {} -> {}'.format(
        num - num_after, num, num_after))
    return filtered_roidb


def load_gt_roidb(dataset_name, image_set_name, root_path, dataset_path,
                  result_path=None, flip=False):
    """ load ground truth roidb """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path,
                              result_path, binary_thresh = 0.4, mask_size = 21)

    roidb = imdb.gt_roidb()
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb

def load_gt_sdsdb(dataset_name, image_set_name, root_path, dataset_path,
                  result_path=None, flip=False):
    """ load ground truth roidb """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path,
                              result_path)
    roidb = imdb.gt_sdsdb()
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb


def load_gt_test_imdb():
    """ load ground truth roidb """
    dataset_name = cfg.dataset.dataset
    image_set_name = cfg.dataset.image_set_test
    root_path = cfg.DATA_DIR
    dataset_path = cfg.dataset.dataset_path
    result_path = None
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path,
                              result_path)
    return imdb

def merge_roidb(roidbs):
    """ roidb are list, concat them together """
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    return roidb


def combined_roidb():
    image_sets = [iset for iset in cfg.dataset.image_set.split('+')]
    roidbs = [load_gt_roidb(
        cfg.dataset.dataset, image_set, cfg.DATA_DIR,
        cfg.dataset.dataset_path, flip=cfg.TRAIN.USE_FLIPPED)
        for image_set in image_sets]

    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb)
    return roidb

def combined_sdsdb():
    image_sets = [iset for iset in cfg.dataset.image_set.split('+')]
    roidbs = [load_gt_sdsdb(
        cfg.dataset.dataset, image_set, cfg.DATA_DIR,
        cfg.dataset.dataset_path, flip=cfg.TRAIN.USE_FLIPPED)
        for image_set in image_sets]

    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb)
    return roidb

if __name__ == '__main__':
    pass
