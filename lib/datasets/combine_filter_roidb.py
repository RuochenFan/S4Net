# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""

from config import cfg
from datasets.factory import get_imdb
import datasets.imdb
import datasets.roi_data_layer.roidb as rdl_roidb
import numpy as np

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        if 'max_overlaps' not in entry.keys():
            from IPython import embed; embed()
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
    print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after))
    return filtered_roidb


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    # print('Preparing training data...')
    # rdl_roidb.prepare_roidb(imdb)
    # print('done')

    return imdb.roidb

def combined_roidb(imdb_names):
    roidbs = []
    for imdb_name in imdb_names.split('+'):
        imdb = get_imdb(imdb_name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        roidbs.append(roidb)
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    imdb = datasets.imdb.imdb(imdb_names, imdb.classes)
    return imdb, roidb

# def combined_roidb(imdb_names):
#     def get_roidb(imdb_name):
#         imdb = get_imdb(imdb_name)
#         print('Loaded dataset `{:s}` for training'.format(imdb.name))
#         imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
#         print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
#         roidb = get_training_roidb(imdb)
#         return roidb
#     roidbs = [get_roidb(s) for s in imdb_names.split('+')]
#     roidb = roidbs[0]
#     if len(roidbs) > 1:
#         for r in roidbs[1:]:
#             roidb.extend(r)
#         tmp = get_imdb(imdb_names.split('+')[1])
#         imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
#     else:
#         imdb = get_imdb(imdb_names)
#     return imdb, roidb


if __name__ == '__main__':
    pass
