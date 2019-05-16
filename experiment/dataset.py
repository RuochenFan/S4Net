# Authors: 644142239@qq.com (Ruochen Fan)

import numpy as np
import os.path as osp
import pickle as pkl

from config import cfg

class Dataset:
    def __init__(self, isTraining=True):
        with open(osp.join(cfg.DATA_DIR, 'dataset.pkl'), 'rb') as f:
            dataset = pkl.load(f)

        if isTraining:
            self.imgs = dataset['train_imgs']
            self.boxes = dataset['train_boxes']
            self.segs = dataset['train_segs']
        else:
            self.imgs = dataset['test_imgs']
            self.boxes = dataset['test_boxes']
            self.segs = dataset['test_segs']
        self.pos = 0
        self.trainset_size = 500
        self.isTraining = isTraining
    def forward(self):
        if self.isTraining:
            img = self.imgs[self.pos % self.trainset_size]
            gt_boxes = self.boxes[self.pos % self.trainset_size]
            gt_masks = self.segs[self.pos % self.trainset_size]
        else:
            img = self.imgs[self.pos%len(self.imgs)]
            gt_boxes = self.boxes[self.pos % len(self.boxes)]
            gt_masks = self.segs[self.pos % len(self.segs)]
        self.pos += 1
        return img, gt_boxes, gt_masks




