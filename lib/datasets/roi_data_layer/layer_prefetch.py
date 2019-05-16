# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""
import time
#from multiprocessing import Process
from multiprocessing import Process, Queue
import threading

import numpy as np
from config import cfg

from datasets.roi_data_layer.minibatch import get_minibatch

class RoIDataLayer(object):
    """Fast R-CNN data layer used for training."""

    def __init__(self, roidb, num_classes, random=False):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._num_classes = num_classes
        # Also set a random flag
        self._random = random
        self._shuffle_roidb_inds()

        if cfg.TRAIN.USE_PREFETCH:
            self.n_iter = 1
            self.data_ready = [threading.Event() for _ in range(self.n_iter)]
            self.data_taken = [threading.Event() for _ in range(self.n_iter)]
            for e in self.data_taken:
                e.set()
            self.started = True
            self.current_batch = [None for _ in range(self.n_iter)]
            self.next_batch = [None for _ in range(self.n_iter)]

            def prefetch_func(self, i):
                """Thread entry"""
                while True:
                    self.data_taken[i].wait()
                    if not self.started:
                        break
                    try:
                        blobs_list = []
                        for gpu_id in range(cfg.num_gpu):
                            db_inds = self._get_next_minibatch_inds()
                            minibatch_db = [self._roidb[db_ind] for db_ind in db_inds]
                            blobs = get_minibatch(
                                minibatch_db, self._num_classes)
                            blobs_list.append(blobs)
                        self.next_batch[i] = blobs_list
                    except StopIteration:
                        self.next_batch[i] = None
                    self.data_taken[i].clear()
                    self.data_ready[i].set()

            self.prefetch_threads = [
                threading.Thread(target=prefetch_func, args=[self, i]) \
                for i in range(self.n_iter)]
            for thread in self.prefetch_threads:
                thread.setDaemon(True)
                thread.start()

    def __del__(self):
        self.started = False
        for e in self.data_taken:
            e.set()
        for thread in self.prefetch_threads:
            thread.join()


    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        # If the random flag is set,
        # then the database is shuffled according to system time
        # Useful for the validation set
        if self._random:
            st0 = np.random.get_state()
            millis = int(round(time.time() * 1000)) % 4294967295
            np.random.seed(millis)

        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))

            #py rfcn random perm
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))

            ''' mxnet random perm
            extra = inds.shape[0] % cfg.TRAIN.IMS_PER_BATCH
            inds_ = np.reshape(inds[:-extra], (-1, cfg.TRAIN.IMS_PER_BATCH))
            row_perm = np.random.permutation(np.arange(inds_.shape[0]))
            inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
            '''
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        # Restore the random state
        if self._random:
            np.random.set_state(st0)

        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""

        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH

        return db_inds

    def iter_next(self):
        for e in self.data_ready:
            e.wait()
        if self.next_batch[0] is None:
            return False
        else:
            self.current_batch = self.next_batch[0]
            for e in self.data_ready:
                e.clear()
            for e in self.data_taken:
                e.set()
            return True

    def forward(self, n_iter=1, is_reshape=True):
        """Get blobs and copy them into this layer's top blob vector."""
        if cfg.TRAIN.USE_PREFETCH:
            if self.iter_next():
                return self.current_batch
            else:
                raise StopIteration
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(
            minibatch_db, self._num_classes, is_reshape=is_reshape)


