# Authors: 644142239@qq.com (Ruochen Fan)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, cv2, glob, os, pprint, sys, time
import numpy as np
import os.path as osp
import random
import setproctitle
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf

from config import cfg
from dataset import Dataset
import network_desp
from tf_utils.model_parallel import average_gradients, sum_gradients
from utils.logger import QuickLogger
from utils.timer import Timer

try:
    import cPickle as pickle
except ImportError:
    import pickle

model_dump_dir = cfg.TRAIN_LOG_ADDR
if not os.path.exists(model_dump_dir):
    os.makedirs(model_dump_dir)


def get_variables_in_checkpoint_file(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print(
                "It's likely that your checkpoint file has been compressed "
                "with SNAPPY.")

def snapshot(sess, saver, iter):
    filename = 'iter_{:d}'.format(iter) + '.ckpt'
    filename = os.path.join(model_dump_dir, filename)
    saver.save(sess, filename)
    print('Wrote snapshot to: {:s}'.format(filename))

    # Also store some meta information, random state, etc.
    nfilename = '_iter_{:d}'.format(iter) + '.pkl'
    nfilename = os.path.join(model_dump_dir, nfilename)
    st0 = np.random.get_state()
    # cur = self.data_layer._cur
    # perm = self.data_layer._perm

    with open(nfilename, 'wb') as fid:
        pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
        #pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
        #pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
        pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

def make_data(image, gt_boxes, segm):
    image = image.astype(np.float32)
    image -= [102.9801, 115.9465, 122.7717]

    segm = segm.astype(np.uint8)
    info = np.zeros([1, 3], dtype=np.float32)
    gt_masks = np.zeros(
        [segm.shape[0], cfg.image_size // cfg.seg_dp_ratio, cfg.image_size // cfg.seg_dp_ratio], dtype=np.uint8)

    original_w = image.shape[1]
    original_h = image.shape[0]
    image = cv2.resize(image, (cfg.image_size, cfg.image_size))
    image = np.stack([image])

    gt_boxes = np.concatenate([gt_boxes, np.ones([len(gt_boxes), 1])], axis=1)
    gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]
    gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]
    gt_boxes[:, 0] *= (cfg.image_size / original_w)
    gt_boxes[:, 1] *= (cfg.image_size / original_h)
    gt_boxes[:, 2] *= (cfg.image_size / original_w)
    gt_boxes[:, 3] *= (cfg.image_size / original_h)

    for i in range(segm.shape[0]):
        gt_masks[i] = cv2.resize(segm[i], (cfg.image_size // cfg.seg_dp_ratio, cfg.image_size // cfg.seg_dp_ratio))

    info[0, 0] = cfg.image_size
    info[0, 1] = cfg.image_size

    return image, info, gt_boxes, gt_masks

def random_flip(image, gt_boxes, gt_masks):
    if random.randint(0,1000)%2 == 0:
        return image, gt_boxes, gt_masks
    image = image[:, :, ::-1, :]
    oldx1 = gt_boxes[:, 0].copy()
    oldx2 = gt_boxes[:, 2].copy()
    gt_boxes[:, 0] = cfg.image_size - oldx2
    gt_boxes[:, 2] = cfg.image_size - oldx1
    gt_masks = gt_masks[:,:,::-1]

    return image, gt_boxes, gt_masks


def train(logger):
    dataset = Dataset(isTraining=True)
    net = network_desp.Network()
    with tf.Graph().as_default(), tf.device('/device:CPU:0'):
        tf.set_random_seed(cfg.RNG_SEED)

        lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        momentum = cfg.TRAIN.MOMENTUM
        opt = tf.train.MomentumOptimizer(lr, momentum)
        # opt = tf.train.AdamOptimizer(lr)
        inputs_list = []
        for i in range(cfg.num_gpu):
            inputs = []
            inputs.append(tf.placeholder(tf.float32, shape=[1, cfg.image_size, cfg.image_size, 3]))
            inputs.append(tf.placeholder(tf.float32, shape=[1, 3]))
            inputs.append(tf.placeholder(tf.float32, shape=[None, 5]))
            inputs.append(tf.placeholder(tf.uint8, shape=[None, None, None]))
            inputs.append(tf.placeholder(tf.float32, shape=[]))
            inputs.append(tf.placeholder(tf.float32, shape=[]))
            inputs_list.append(inputs)

        tower_grads = []
        biases_regularizer = tf.no_regularizer
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(cfg.num_gpu):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as scope:
                        with slim.arg_scope([slim.model_variable,
                                             slim.variable],
                                            device='/device:CPU:0'):
                            with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                                            slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                                            weights_regularizer=weights_regularizer,
                                            biases_regularizer=biases_regularizer,
                                            biases_initializer=tf.constant_initializer(0.0)):
                                loss = net.inference('TRAIN', inputs_list[i])
                                loss = loss * (1.0 / cfg.num_gpu)
                        tf.get_variable_scope().reuse_variables()
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)

        if len(tower_grads) > 1:
            grads = sum_gradients(tower_grads)
        else:
            grads = tower_grads[0]
        #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        # grads = [(tf.clip_by_value(grad, -0.01, 0.01), var) for grad, var in grads]
        apply_gradient_op = opt.apply_gradients(grads)

        saver = tf.train.Saver(max_to_keep=100000)
        tfconfig = tf.ConfigProto(allow_soft_placement=True,
            log_device_placement=False)
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)

        variables = tf.global_variables()
        var_keep_dic = get_variables_in_checkpoint_file(cfg.weight)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE))
        sess.run(tf.variables_initializer(variables, name='init'))

        variables_to_restore = []
        for v in variables:
            if v.name.split(':')[0] in var_keep_dic:
                # print('Varibles restored: %s' % v.name)
                variables_to_restore.append(v)

        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, cfg.weight)

        timer = Timer()
        train_collection = net.get_train_collection()
        sess_ret = []
        sess_ret.append(apply_gradient_op)
        for col in train_collection.values():
            sess_ret.append(col)
        sess_ret.append(lr)

        summary_writer = tf.summary.FileWriter(model_dump_dir, sess.graph)
        sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE))
        for iter in range(1, cfg.TRAIN.MAX_ITER+1):
            if iter == cfg.TRAIN.STEPSIZE:
                sess.run(
                    tf.assign(lr, cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA))
            if iter == cfg.TRAIN.STEPSIZE_2:
                sess.run(
                    tf.assign(lr, cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA * cfg.TRAIN.GAMMA))
            feed_dict = {}
            for inputs in inputs_list:

                image, gt_boxes, gt_masks = dataset.forward()
                image, info, gt_boxes, gt_masks = make_data(image, gt_boxes, gt_masks)
                image, gt_boxes, gt_masks = random_flip(image, gt_boxes, gt_masks)
                feed_dict[inputs[0]] = image
                feed_dict[inputs[1]] = info
                feed_dict[inputs[2]] = gt_boxes
                feed_dict[inputs[3]] = gt_masks
                if iter < 5000:
                    feed_dict[inputs[4]] = 1
                else:
                    feed_dict[inputs[4]] = 1
                feed_dict[inputs[5]] = iter

            timer.tic()
            _, rpn_loss_box, rpn_loss_cat, rpn_cross_entropy_pos, rpn_cross_entropy_neg, loss_seg, loss_wd, total_loss, summuries_str,  \
            cur_lr = sess.run(sess_ret,
                              feed_dict=feed_dict)
            timer.toc()

            if iter % (cfg.TRAIN.DISPLAY) == 0:
                logger.info(
                    'iter: %d/%d, loss: %.4f, box: %.4f,'
                    'cat: %.4f, pos: %.4f, neg: %.4f, seg:%.3f, wd: %.4f, lr: %.4f, speed: %.3fs/iter' % \
                    (iter, cfg.TRAIN.MAX_ITER, total_loss, rpn_loss_box, rpn_loss_cat, rpn_cross_entropy_pos, rpn_cross_entropy_neg, loss_seg, loss_wd, cur_lr,
                     timer.average_time))

            if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                snapshot(sess, saver, iter)
                setproctitle.setproctitle(
                    'train ' + os.path.split(os.path.realpath(__file__))[0] +
                    ' iter:' + str(iter) + " of " + str(cfg.TRAIN.MAX_ITER))
            if iter % 100 == 0:
                summary_writer.add_summary(summuries_str, iter)

if __name__ == '__main__':
    setproctitle.setproctitle('train ' +
                              os.path.split(os.path.realpath(__file__))[0])
    train_logger = QuickLogger(cfg.TRAIN_LOG_ADDR).get_logger()
    train_logger.info(cfg)
    np.random.seed(cfg.RNG_SEED)

    train(train_logger)
