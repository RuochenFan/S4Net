# Authors: 644142239@qq.com (Ruochen Fan)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
import sys


cfg = edict()

train_gpu_ids="0"

cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))


cfg.DATA_DIR = osp.abspath(osp.join(cfg.ROOT_DIR, 'data'))
cfg.TEST_LOG_ADDR = osp.abspath(osp.join(cfg.ROOT_DIR, 'logs/test_logs'))
cfg.TRAIN_LOG_ADDR = osp.abspath(osp.join(cfg.ROOT_DIR, 'logs/train_logs'))
cfg.weight = osp.join(cfg.ROOT_DIR, 'data/resnet_v1_50.ckpt')  #fpn_base

cfg.image_size = 320
cfg.stride = [16]
cfg.anchor_scales = [4, 8, 16, 32]
cfg.anchor_ratios = [0.5, 1, 2]
cfg.simga_rpn = 3
cfg.seg_dp_ratio = 8

cfg.DEDUP_BOXES = 1. / 16.
cfg.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
cfg.RNG_SEED = 3
cfg.EPS = 1e-14

cfg.num_gpu = len(train_gpu_ids.split(','))
##################TRAIN config####################################
cfg.TRAIN = edict()
cfg.TRAIN.STEPSIZE = 20000 // cfg.num_gpu
cfg.TRAIN.STEPSIZE_2 = 30000 // cfg.num_gpu
cfg.TRAIN.MAX_ITER = 40000 // cfg.num_gpu
cfg.TRAIN.LEARNING_RATE = 0.002 * cfg.num_gpu
cfg.TRAIN.SNAPSHOT_ITERS = 10000 // cfg.num_gpu

cfg.TRAIN.HAS_RPN = True
cfg.TRAIN.IMS_PER_BATCH = 1
cfg.TRAIN.PROPOSAL_METHOD = 'gt'
cfg.TRAIN.DISPLAY = 50
cfg.TRAIN.DOUBLE_BIAS = False

cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.GAMMA = 0.1
cfg.TRAIN.BIAS_DECAY = False
cfg.TRAIN.USE_GT = True
cfg.TRAIN.TRUNCATED = False
cfg.TRAIN.ASPECT_GROUPING = True
cfg.TRAIN.SUMMARY_INTERVAL = 180

cfg.TRAIN.SCALES = (600,)
cfg.TRAIN.MAX_SIZE = 1000

cfg.TRAIN.BATCH_SIZE = 256
cfg.TRAIN.FG_FRACTION = 0.25
cfg.TRAIN.FG_THRESH = 0.5
cfg.TRAIN.BG_THRESH_HI = 0.5
cfg.TRAIN.BG_THRESH_LO = 0.0
cfg.TRAIN.USE_FLIPPED = True
cfg.TRAIN.BBOX_REG = True
cfg.TRAIN.BBOX_THRESH = 0.5

cfg.TRAIN.BBOX_NORMALIZE_TARGETS = True
cfg.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
cfg.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
cfg.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

cfg.TRAIN.RPN_OVERLAP = 0.4
cfg.TRAIN.RPN_SEG_OVERLAP = 0.5
cfg.TRAIN.RPN_CLOBBER_POSITIVES = False
cfg.TRAIN.RPN_FG_FRACTION = 0.5
cfg.TRAIN.RPN_BATCHSIZE = 256
cfg.TRAIN.RPN_NMS_THRESH = 0.4
cfg.TRAIN.RPN_PRE_NMS_TOP_N = 5000
cfg.TRAIN.RPN_POST_NMS_TOP_N = 20
# cfg.TRAIN.RPN_MIN_SIZE = 16
cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
cfg.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
cfg.TRAIN.USE_ALL_GT = True

cfg.RESNET = edict()
cfg.RESNET.MAX_POOL = False
cfg.RESNET.BN_TRAIN = False

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

lib_path = osp.join(cfg.ROOT_DIR, 'lib')
add_path(lib_path)

if __name__ == '__main__':
    from IPython import embed; embed()

