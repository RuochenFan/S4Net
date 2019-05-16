# Authors: 644142239@qq.com (Ruochen Fan)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
from config import cfg
import cv2
from IPython import embed
import numpy as np
import os
import os.path as osp
import pickle as pkl
import sys
import tensorflow as tf

from dataset import Dataset
from evaluation import eval
import network_desp
from utils.cython_nms import nms, nms_new
from utils.logger import QuickLogger
from utils.timer import Timer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_addr',
                        dest='weights_addr',
                        default="../logs/iter_40000.ckpt",
                        type=str)
    parser.add_argument('--gpu',
                        dest='gpu',
                        default="0",
                        type=str)
    parser.add_argument('--show',
                        dest='show',
                        default=False,
                        type=bool)
    args = parser.parse_args()
    return args


color_table = np.array([[162, 109, 35], [69, 94, 183], [72, 161, 198],
                        [82, 158, 127], [120, 72, 122], [105, 124, 135]]*10)

def clip_boxes(boxes, im_shape):
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def im_detect(sess, net, inputs, im, test_collect, _t):
    image = im.astype(np.float32, copy=True)
    image -= cfg.PIXEL_MEANS


    image = image[:,:,::-1]
    new_size = cfg.image_size
    original_w = image.shape[1]
    original_h = image.shape[0]
    image = cv2.resize(image, (new_size, new_size))
    image = np.stack([image])
    info = np.array([[image.shape[1], image.shape[2], 3]])
    feed_dict = {inputs[0]: image,
                 inputs[1]: info}

    _t['inference'].tic()
    cat_prob, bbox_pred, anchors, seg_pred = sess.run(
        test_collect, feed_dict=feed_dict)
    cat_prob = cat_prob.reshape([-1, 2])
    seg_pred = softmax(seg_pred)
    _t['inference'].toc()



    bboxes = bbox_pred.reshape((-1, 4))
    bboxes[:, 0] *= (original_w/new_size)
    bboxes[:, 1] *= (original_h/new_size)
    bboxes[:, 2] *= (original_w/new_size)
    bboxes[:, 3] *= (original_h/new_size)
    bboxes = clip_boxes(bboxes, im.shape) # (-1, 4) from (H, W, A)

    masks = np.zeros([bboxes.shape[0], original_h, original_w], dtype=np.float32)
    for i in range(bboxes.shape[0]):
        x0 = int(bboxes[i, 0])
        y0 = int(bboxes[i, 1])
        x1 = int(bboxes[i, 2])+1
        y1 = int(bboxes[i, 3])+1
        masks[i, y0:y1, x0:x1] = 1.

    if args.show:
        max_cat_prob = np.max(cat_prob[:, 1:], axis=1)
        keep = max_cat_prob > 0.4
        cat_prob = cat_prob[keep]
        bboxes = bboxes[keep]
        seg_pred = seg_pred[keep]
        masks = masks[keep]

    return cat_prob, bboxes, seg_pred, masks


def test_net(args):
    log_name = 'test.logs'
    logger = QuickLogger(log_dir=cfg.TEST_LOG_ADDR, log_name=log_name).get_logger()

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    inputs = []
    inputs.append(tf.placeholder(tf.float32, shape=[1, None, None, 3]))
    inputs.append(tf.placeholder(tf.float32, shape=[1, 3]))
    inputs.append(tf.placeholder(tf.float32, shape=[None, 5]))
    inputs.append(tf.placeholder(tf.uint8, shape=[None, None, None]))
    inputs.append(tf.placeholder(tf.float32, shape=[]))
    inputs.append(tf.placeholder(tf.float32, shape=[]))

    sess = tf.Session(config=tfconfig)
    net = network_desp.Network()
    net.inference('TEST', inputs)
    test_collect_dict = net.get_test_collection()
    test_collect = [it for it in test_collect_dict.values()]

    weights_filename = osp.join(args.weights_addr)
    saver = tf.train.Saver()
    saver.restore(sess, weights_filename)
    np.random.seed(cfg.RNG_SEED)

    dataset = Dataset(isTraining=False)

    num_images = 300

    res = []

    _t = {'inference': Timer(), 'im_detect': Timer(), 'misc': Timer()}
    for i in range(num_images):
        im, gt_boxes, gt_masks = dataset.forward()

        _t['im_detect'].tic()
        cat_prob, boxes, seg_pred, masks = im_detect(sess, net, inputs, im, test_collect, _t)
        _t['im_detect'].toc()

        _t['misc'].tic()


        cls_scores = cat_prob[:, 1]
        cls_dets = np.hstack((boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)

        segmaps = np.zeros([len(seg_pred), im.shape[0], im.shape[1]])
        img_for_show = copy.deepcopy(im)
        for k in range(len(seg_pred)):
            img_for_single_instance = copy.deepcopy(im)

            segmap = seg_pred[k, :, :, 1]
            segmap = cv2.resize(segmap, (img_for_single_instance.shape[1], img_for_single_instance.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            segmap_masked = segmap * masks[k]
            segmaps[k] = segmap_masked
            if args.show:
                color = color_table[k]
                img_for_show[segmap_masked > 0.5] = 0.8 * color + 0.2 * img_for_show[segmap_masked > 0.5]

        res.append({'gt_masks':gt_masks, 'segmaps':segmaps, 'scores':cls_scores, 'img':im})

        _t['misc'].toc()
        if args.show:
            if len(cls_dets) > 0:
              for record in cls_dets:
                x0 = record[0]
                y0 = record[1]
                x1 = record[2]
                y1 = record[3]
                cv2.rectangle(img_for_show, (int(x0), int(y0)), (int(x1), int(y1)), (73, 196, 141), 2)

            gt = np.zeros([gt_masks[0].shape[0], gt_masks[0].shape[1], 3], gt_masks[0].dtype)
            for k in range(len(gt_masks)):
                mask = gt_masks[k].reshape([gt_masks[k].shape[0], gt_masks[k].shape[1], 1])
                gt += mask * color_table[k].reshape([1,1,3])

            cv2.imwrite('seg.jpg', img_for_show)
            cv2.imwrite('img.jpg', im)
            cv2.imwrite('gt.jpg', gt)
            input('Drawn')

        logger.info('im_detect: {:d}/{:d} {:.3f}s {:.3f}s {:.3f}s' \
                    .format(i + 1, num_images, _t['inference'].average_time, _t['im_detect'].average_time,
                            _t['misc'].average_time))

    logger.info('Evaluating detections')

    map07 = eval(res, 0.7)
    map05 = eval(res, 0.5)

    logger.info('mAP07:%f mAP05:%f' % (map07, map05))


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    test_net(args)
