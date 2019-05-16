# Authors: 644142239@qq.com (Ruochen Fan)

import copy, cv2
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import regularizers, \
    initializers, layers

from anchor_target_layer import anchor_target_layer
from collections import OrderedDict as dict
from config import cfg
from detection_opr.rpn.snippets import generate_anchors_pre
from detection_opr.rpn.proposal_target_layer import proposal_target_layer
from detection_opr.utils.bbox_transform import bbox_transform_inv, clip_boxes
from detection_opr.utils import loss_opr
from detection_opr.common_opr.faster_rcnn_opr import crop_pool_layer, \
    reshape_layer
import resnet_utils, resnet_v1
from utils.cython_bbox import bbox_overlaps

def resnet_arg_scope(is_training=False,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': cfg.RESNET.BN_TRAIN,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }

    with arg_scope(
            [slim.conv2d],
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            weights_initializer=initializers.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=nn_ops.relu,
            normalizer_fn=layers.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


def global_context_module(bottom, prefix='', ks=15, chl_mid=256, chl_out=1024):
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    col_max = slim.conv2d(bottom, chl_mid, [ks, 1], 
        trainable=True, activation_fn=None,
        weights_initializer=initializer, scope=prefix + '_conv%d_w_pre' % ks)
    col = slim.conv2d(col_max, chl_out, [1, ks], 
        trainable=True, activation_fn=None,
        weights_initializer=initializer, scope=prefix + '_conv%d_w' % ks)
    row_max = slim.conv2d(bottom, chl_mid, [1, ks], 
        trainable=True, activation_fn=None,
        weights_initializer=initializer, scope=prefix + '_conv%d_h_pre' % ks)
    row = slim.conv2d(row_max, chl_out, [ks, 1], 
        trainable=True, activation_fn=None,
        weights_initializer=initializer, scope=prefix + '_conv%d_h' % ks)
    s = row + col
    return s

def softmax_loss_ohem(cls_score, label, nr_ohem_sampling):
    cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=cls_score, labels=label)
    topk_val, topk_idx = tf.nn.top_k(cls_loss, k=nr_ohem_sampling,
        sorted=False, name='ohem_cls_loss_index')
    cls_loss_ohem = tf.gather(cls_loss, topk_idx, name='ohem_cls_loss')
    cls_loss_ohem = tf.reduce_sum(cls_loss_ohem) / nr_ohem_sampling
    return cls_loss_ohem

def _smooth_l1_loss_base(bbox_pred, bbox_targets, bbox_inside_weights,
                         bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(
        tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    return out_loss_box

def smooth_l1_loss_valid(bbox_pred, bbox_targets, bbox_inside_weights,
                         bbox_outside_weights, label,
                         background=0, ignore_label=-1,
                         sigma=1.0, dim=[1]):
    value = _smooth_l1_loss_base(bbox_pred, bbox_targets, bbox_inside_weights,
                                 bbox_outside_weights, sigma, dim=[1])
    loss = tf.reduce_sum(value) / 4
    return loss

def generate_seg_gt(bbox_pred, gt_boxes, gt_masks):
    seg_label = np.zeros([len(bbox_pred), cfg.image_size//cfg.seg_dp_ratio, cfg.image_size//cfg.seg_dp_ratio, 2], np.bool)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(bbox_pred, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    for i in range(len(bbox_pred)):
        id = argmax_overlaps[i]
        seg_label[i, :, :, 1] = np.round(gt_masks[id])
        seg_label[i, :, :, 0] = 1 - seg_label[i, :, :, 1]
    seg_label = seg_label.astype(np.int32)
    return seg_label

def proposal_without_nms_layer(rpn_cls_prob_fg, rpn_bbox_pred, im_info, anchors):

    pre_nms_topN = cfg.TRAIN.RPN_PRE_NMS_TOP_N
    im_info = im_info[0]

    scores = rpn_cls_prob_fg
    scores = scores.reshape((-1, 1))
    rpn_bbox_pred[:, 2:4] = np.minimum(20, rpn_bbox_pred[:, 2:4])
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
    proposals = clip_boxes(proposals, im_info[:2])

    # Pick the top region proposals
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order].flatten()

    ##why add one, because tf nms assume x2,y2 does not include border
    proposals_addone = np.array(proposals)
    proposals_addone[:, 2] += 1
    proposals_addone[:, 3] += 1
    return proposals, scores, proposals_addone, order

def generate_masks(bbox_pred):
    rois = bbox_pred.copy()
    masks = np.zeros([rois.shape[0], cfg.image_size//cfg.seg_dp_ratio, cfg.image_size//cfg.seg_dp_ratio, 1], dtype=np.float32)
    rois //= cfg.seg_dp_ratio
    for i in range(rois.shape[0]):
        x0 = int(rois[i, 0])
        y0 = int(rois[i, 1])
        x1 = int(rois[i, 2])+1
        y1 = int(rois[i, 3])+1
        masks[i, y0:y1, x0:x1, 0] = 1.
    return masks

def generate_bimasks(bbox_pred):
    rois = bbox_pred.copy()
    masks = np.zeros([rois.shape[0], cfg.image_size//cfg.seg_dp_ratio, cfg.image_size//cfg.seg_dp_ratio, 1], dtype=np.float32)
    bimasks = np.zeros([rois.shape[0], cfg.image_size // cfg.seg_dp_ratio, cfg.image_size // cfg.seg_dp_ratio, 1],
                     dtype=np.float32)
    rois //= cfg.seg_dp_ratio
    for i in range(rois.shape[0]):
        x0 = int(rois[i, 0])
        y0 = int(rois[i, 1])
        x1 = int(rois[i, 2])+1
        y1 = int(rois[i, 3])+1
        masks[i, y0:y1, x0:x1, 0] = 1.
    for i in range(rois.shape[0]):
        x0 = int(rois[i, 0])
        y0 = int(rois[i, 1])
        x1 = int(rois[i, 2])+1
        y1 = int(rois[i, 3])+1
        w = x1-x0
        h = y1-y0
        bimasks[i, (y0-np.round(h//3)):(y1+np.round(h//3)), (x0-np.round(w//3)):(x1+np.round(w//3)), 0] = -1.
        bimasks[i, y0:y1, x0:x1, 0] = 1.

    return masks, bimasks


class Network(object):
    def __init__(self):
        pass

    def inference(self, mode, inputs):
        is_training = mode == 'TRAIN'

        ###decode your inputs
        image = inputs[0]
        im_info = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]
        seg_loss_gate = inputs[4]
        iter = inputs[5]
        image.set_shape([1, None, None, 3])
        im_info.set_shape([1, 3])
        if mode == 'TRAIN':
            gt_boxes.set_shape([None, 5])
        ##end of decode

        num_anchors = len(cfg.anchor_scales) * len(cfg.anchor_ratios)
        bottleneck = resnet_v1.bottleneck
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

        blocks = [
            resnet_utils.Block('block1', bottleneck,
                               [(256, 64, 1, 1)] * 2 + [(256, 64, 2, 1)]),
            resnet_utils.Block('block2', bottleneck,
                               [(512, 128, 1, 1)] * 3 + [(512, 128, 2, 1)]),
            resnet_utils.Block('block3', bottleneck,
                               [(1024, 256, 1, 1)] * 5 + [(1024, 256, 2, 1)]),
            resnet_utils.Block('block4', bottleneck, [(2048, 512, 1, 1)] * 3)
        ]

        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            with tf.variable_scope('resnet_v1_50', 'resnet_v1_50'):
                net = resnet_utils.conv2d_same(
                    image, 64, 7, stride=2, scope='conv1')
                net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
                net = slim.max_pool2d(
                    net, [3, 3], stride=2, padding='VALID', scope='pool1')
            net, _ = resnet_v1.resnet_v1(
                net, blocks[0:1],
                global_pool=False, include_root_block=False,
                scope='resnet_v1_50')

        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            net2, _ = resnet_v1.resnet_v1(
                net, blocks[1:2],
                global_pool=False, include_root_block=False,
                scope='resnet_v1_50')
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            net3, _ = resnet_v1.resnet_v1(
                net2, blocks[2:3],
                global_pool=False, include_root_block=False,
                scope='resnet_v1_50')
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            net4, _ = resnet_v1.resnet_v1(
                net3, blocks[3:4],
                global_pool=False, include_root_block=False,
                scope='resnet_v1_50')

        namescope = tf.no_op(name='.').name[:-1]
        resnet_features_name = [
            namescope+'resnet_v1_50_1/block1/unit_2/bottleneck_v1/Relu:0',
            namescope+'resnet_v1_50_2/block2/unit_3/bottleneck_v1/Relu:0',
            namescope+'resnet_v1_50_3/block3/unit_5/bottleneck_v1/Relu:0',
            namescope+'resnet_v1_50_4/block4/unit_3/bottleneck_v1/Relu:0'
        ]

        resnet_features = []
        for i in range(len(resnet_features_name)):
            resnet_features.append(tf.get_default_graph().get_tensor_by_name(resnet_features_name[i]))

        mid_channels = 256

        with tf.variable_scope('resnet_v1_50', 'resnet_v1_50', regularizer=tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)):
            finer = slim.conv2d(resnet_features[-1], mid_channels, [1, 1], trainable=is_training, weights_initializer=initializer, activation_fn=None, scope='pyramid/res5')
            pyramid_features = [finer]
            for i in range(4, 1, -1):
                lateral = slim.conv2d(resnet_features[i-2], mid_channels, [1, 1], trainable=is_training, weights_initializer=initializer, activation_fn=None, scope='lateral/res{}'.format(i))
                upsample = tf.image.resize_bilinear(finer, (tf.shape(lateral)[1], tf.shape(lateral)[2]), name='upsample/res{}'.format(i))
                finer = upsample + lateral
                pyramid = slim.conv2d(finer, mid_channels, [3, 3], trainable=is_training, weights_initializer=initializer, activation_fn=None, scope='pyramid/res{}'.format(i))
                pyramid_features.append(pyramid)
            pyramid_features.reverse()
            pyramid = slim.avg_pool2d(pyramid_features[-1], [2, 2], stride=2, padding='SAME', scope='pyramid/res6')
            pyramid_features.append(pyramid)
        # pyramid_features downsampling rate:   4, 8, 16, 32, 64

        allowed_borders = [16, 32, 64, 128, 256]
        feat_strides = np.array([4, 8, 16, 32, 64])
        anchor_scaleses = np.array([[1], [2], [4], [8], [16]])

        with tf.variable_scope('resnet_v1_50', 'resnet_v1_50',
                               regularizer=tf.contrib.layers.l2_regularizer(
                                   cfg.TRAIN.WEIGHT_DECAY)) as scope:
            num_anchors = len(cfg.anchor_ratios)
            rpn_cls_prob_pyramid = []
            rpn_bbox_pred_pyramid = []
            anchors_pyramid = []
            rpn_cls_score_reshape_pyramid = []


            rpn_label_pyramid = []
            labels_cat_pyramid = []
            rpn_bbox_targets_pyramid = []
            rpn_bbox_inside_weights_pyramid = []
            rpn_bbox_outside_weights_pyramid = []

            with tf.variable_scope('resnet_v1_50_rpn', 'resnet_v1_50_rpn') as scope:
                for i, pyramid_feature in enumerate(pyramid_features):
                    with tf.variable_scope('anchor/res{}'.format(i+2)):
                        shape = tf.shape(pyramid_feature)
                        height, width = shape[1], shape[2]
                        anchors, _ = tf.py_func(
                            generate_anchors_pre,
                            [height, width, feat_strides[i], anchor_scaleses[i], cfg.anchor_ratios],
                            [tf.float32, tf.int32])

                    # rpn
                    rpn = slim.conv2d(pyramid_feature, 512, [3, 3], trainable=is_training, weights_initializer=initializer, activation_fn=nn_ops.relu, scope='rpn_conv')
                    # head
                    rpn_cls_score = slim.conv2d(rpn, num_anchors * 2, [3, 3], trainable=is_training, weights_initializer=initializer, activation_fn=None, scope='rpn_cls_score')
                    rpn_cls_score_reshape = tf.reshape(rpn_cls_score, [-1, 2], name='rpn_cls_score_reshape/res{}'.format(i + 2))
                    rpn_cls_prob = tf.nn.softmax(rpn_cls_score_reshape, name="rpn_cls_prob_reshape/res{}".format(i + 2))
                    rpn_bbox_pred = slim.conv2d(rpn, num_anchors * 4, [3, 3], trainable=is_training, weights_initializer=initializer, activation_fn=None, scope='rpn_bbox_pred')
                    rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])

                    # share rpn
                    scope.reuse_variables()

                    rpn_cls_prob_pyramid.append(rpn_cls_prob)
                    rpn_bbox_pred_pyramid.append(rpn_bbox_pred)
                    anchors_pyramid.append(anchors)
                    rpn_cls_score_reshape_pyramid.append(rpn_cls_score_reshape)

                    if is_training:
                        with tf.variable_scope('anchors_targets/res{}'.format(i + 2)):
                            rpn_labels, rpn_bbox_targets, \
                            rpn_bbox_inside_weights, rpn_bbox_outside_weights, labels_cat, gt_id = \
                                tf.py_func(
                                    anchor_target_layer,
                                    [rpn_cls_score, gt_boxes, im_info,
                                     feat_strides[i], anchors, num_anchors, gt_masks],
                                    [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int64])
                            rpn_labels = tf.to_int32(rpn_labels, name="to_int32") # (1, H, W, A)
                            labels_cat = tf.to_int32(labels_cat, name="to_int32") # (1, H, W, A)

                            rpn_labels = tf.reshape(rpn_labels, [-1])
                            labels_cat = tf.reshape(labels_cat, [-1])
                            rpn_bbox_targets = tf.reshape(rpn_bbox_targets, [-1,4])
                            rpn_bbox_inside_weights = tf.reshape(rpn_bbox_inside_weights, [-1,4])
                            rpn_bbox_outside_weights = tf.reshape(rpn_bbox_outside_weights, [-1,4])

                        rpn_label_pyramid.append(rpn_labels)
                        labels_cat_pyramid.append(labels_cat)
                        rpn_bbox_targets_pyramid.append(rpn_bbox_targets)
                        rpn_bbox_inside_weights_pyramid.append(rpn_bbox_inside_weights)
                        rpn_bbox_outside_weights_pyramid.append(rpn_bbox_outside_weights)

            rpn_cls_prob_pyramid = tf.concat(axis=0, values=rpn_cls_prob_pyramid)
            rpn_bbox_pred_pyramid = tf.concat(axis=0, values=rpn_bbox_pred_pyramid)
            anchors_pyramid = tf.concat(axis=0, values=anchors_pyramid)
            rpn_cls_score_reshape_pyramid = tf.concat(axis=0, values=rpn_cls_score_reshape_pyramid)

        with tf.variable_scope('rois') as scope:
            rpn_cls_prob_bg = rpn_cls_prob_pyramid[:, 0]
            rpn_cls_prob_fg = 1 - rpn_cls_prob_bg

            rpn_proposals, rpn_proposal_scores, \
            rpn_proposals_addone, keep_pre = tf.py_func(
                proposal_without_nms_layer,
                [rpn_cls_prob_fg, rpn_bbox_pred_pyramid,
                 im_info, anchors_pyramid],
                [tf.float32, tf.float32, tf.float32, tf.int64])

            rpn_cls_prob_pyramid = tf.gather(rpn_cls_prob_pyramid, keep_pre)

            keep = tf.image.non_max_suppression(
                rpn_proposals_addone,
                rpn_proposal_scores,
                cfg.TRAIN.RPN_POST_NMS_TOP_N,
                iou_threshold=cfg.TRAIN.RPN_NMS_THRESH)
            bbox_pred = tf.gather(rpn_proposals, keep)
            roi_scores = tf.gather(rpn_proposal_scores, keep)
            anchors_pyramid = tf.gather(anchors_pyramid, keep)
            rpn_cls_prob_pyramid = tf.gather(rpn_cls_prob_pyramid, keep)

        with tf.variable_scope('seg', 'seg', regularizer=tf.contrib.layers.l2_regularizer(
                                   cfg.TRAIN.WEIGHT_DECAY)):
            x = pyramid_features[1]
            seg_pred = slim.conv2d(
                x, 128, [3, 3], trainable=is_training,
                weights_initializer=initializer, scope="pixel_seg_conv_1")
            # br = slim.conv2d(
            #     x, 256, [3, 3], trainable=is_training,
            #     weights_initializer=initializer, scope="pixel_seg_conv_1")
            # br = slim.conv2d(
            #     br, 256, [3, 3], trainable=is_training,
            #     weights_initializer=initializer, scope="pixel_seg_conv_2")
            # x += br
            # br = slim.conv2d(
            #     x, 256, [3, 3], trainable=is_training,
            #     weights_initializer=initializer, scope="pixel_seg_conv_3")
            # br = slim.conv2d(
            #     br, 256, [3, 3], trainable=is_training,
            #     weights_initializer=initializer, scope="pixel_seg_conv_4")
            # x += br
            # br = slim.conv2d(
            #     x, 256, [3, 3], trainable=is_training,
            #     weights_initializer=initializer, scope="pixel_seg_conv_5")
            # br = slim.conv2d(
            #     br, 256, [3, 3], trainable=is_training,
            #     weights_initializer=initializer, scope="pixel_seg_conv_6")
            # x += br

            # x = slim.conv2d_transpose(x, 256, [3, 3], [2, 2], "SAME", scope="pixel_seg_deconv_1")


            if is_training:
                # bbox_pred_seg = tf.concat([bbox_pred, gt_boxes[:, :4]], axis=0)
                bbox_pred_seg = gt_boxes[:, :4]
            else:
                bbox_pred_seg = bbox_pred
            num_proposals = tf.shape(bbox_pred_seg)[0]
            num_proposals = tf.stack([num_proposals])
            one = tf.constant([1], dtype=tf.int32)


            seg_pred_pyramid = tf.tile(seg_pred, tf.concat([num_proposals, one, one, one], axis=0))
            masks, bimasks = tf.py_func(generate_bimasks, [bbox_pred_seg], [tf.float32, tf.float32])
            masks.set_shape([None, None, None, None])
            masks = tf.stop_gradient(masks)
            bimasks.set_shape([None, None, None, None])
            bimasks = tf.stop_gradient(bimasks)

            seg_pred_pyramid = seg_pred_pyramid * bimasks
            x = seg_pred_pyramid

            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

            br = slim.conv2d(
                x, 128, [3, 3], trainable=is_training,
                weights_initializer=initializer, scope="final_conv_1")
            br = slim.conv2d(
                br, 128, [3, 3], trainable=is_training,
                weights_initializer=initializer, scope="final_conv_2")
            x += br

            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

            x = slim.conv2d(
                x, 64, [3, 3], rate=2, trainable=is_training,
                weights_initializer=initializer, scope="final_conv_3")

            br = slim.conv2d(
                x, 64, [3, 3], trainable=is_training,
                weights_initializer=initializer, scope="final_conv_4")
            br = slim.conv2d(
                br, 64, [3, 3], trainable=is_training,
                weights_initializer=initializer, scope="final_conv_5")
            x += br

            # x = tf.image.resize_bilinear(x, (40, 40))

            seg_pred_pyramid = slim.conv2d(
                x, 2, [3, 3], trainable=is_training,
                weights_initializer=initializer, scope="final_conv_6")

            if is_training:
                labels_seg, = tf.py_func(generate_seg_gt, [bbox_pred_seg, gt_boxes, gt_masks], [tf.int32])


        if is_training:
            rpn_label_pyramid = tf.concat(axis=0, values=rpn_label_pyramid)
            labels_cat_pyramid = tf.concat(axis=0, values=labels_cat_pyramid)
            rpn_bbox_targets_pyramid = tf.concat(axis=0, values=rpn_bbox_targets_pyramid)
            rpn_bbox_inside_weights_pyramid = tf.concat(axis=0, values=rpn_bbox_inside_weights_pyramid)
            rpn_bbox_outside_weights_pyramid = tf.concat(axis=0, values=rpn_bbox_outside_weights_pyramid)



        ##############add prediction#####################
        tf.add_to_collection("rpn_cls_prob", rpn_cls_prob_pyramid)
        tf.add_to_collection("rpn_bbox_pred", bbox_pred)
        tf.add_to_collection("anchors", anchors_pyramid)
        tf.add_to_collection("seg_pred_pyramid", seg_pred_pyramid)


        if is_training:
            with tf.variable_scope('loss') as scope:
                #############rpn loss################
                rpn_cls_score = rpn_cls_score_reshape_pyramid
                rpn_label = rpn_label_pyramid
                rpn_select = tf.where(tf.not_equal(rpn_label, -1))
                rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select),
                                           [-1, 2])
                labels_cat = labels_cat_pyramid
                labels_cat = tf.reshape(tf.gather(labels_cat, rpn_select), [-1])


                inds_pos = tf.where(tf.not_equal(labels_cat, 0))
                inds_neg = tf.where(tf.equal(labels_cat, 0))

                rpn_cls_score_pos = tf.reshape(tf.gather(rpn_cls_score, inds_pos),
                                           [-1, 2])
                rpn_cls_score_neg = tf.reshape(tf.gather(rpn_cls_score, inds_neg),
                                               [-1, 2])
                labels_cat_pos = tf.reshape(tf.gather(labels_cat, inds_pos), [-1])
                labels_cat_neg = tf.reshape(tf.gather(labels_cat, inds_neg), [-1])

                rpn_cross_entropy_pos = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=rpn_cls_score_pos, labels=labels_cat_pos))
                rpn_cross_entropy_neg = softmax_loss_ohem(rpn_cls_score_neg, labels_cat_neg, 256)

                rpn_cross_entropy_pos *= 0.3
                rpn_cross_entropy_neg *= 0.3

                bPos = tf.shape(inds_pos)[0] > 0
                zero = tf.constant(0.)
                rpn_cross_entropy_pos = tf.cond(bPos, lambda : rpn_cross_entropy_pos, lambda : zero)

                masks = masks[:, :, :, 0]

                seg_loss = tf.nn.softmax_cross_entropy_with_logits(
                        logits=seg_pred_pyramid, labels=labels_seg)

                seg_loss *= masks
                sum_mask = tf.reduce_sum(masks)
                bPos = sum_mask > 1

                seg_loss = tf.reduce_sum(seg_loss) / sum_mask
                # seg_loss = tf.cond(bPos, lambda: seg_loss, lambda: zero)
                # seg_loss *= seg_loss_gate

                rpn_cross_entropy = rpn_cross_entropy_pos + rpn_cross_entropy_neg



                rpn_loss_box = smooth_l1_loss_valid(
                    rpn_bbox_pred_pyramid, rpn_bbox_targets_pyramid, rpn_bbox_inside_weights_pyramid,
                    rpn_bbox_outside_weights_pyramid, labels_cat_pyramid,
                    sigma=cfg.simga_rpn, dim=[0])

                loss_wd = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

                loss = rpn_cross_entropy + rpn_loss_box + seg_loss + loss_wd

                tf.add_to_collection('rpn_cross_entropy_pos', rpn_cross_entropy_pos)
                tf.add_to_collection('rpn_cross_entropy_neg', rpn_cross_entropy_neg)
                tf.add_to_collection('rpn_cross_entropy', rpn_cross_entropy)
                tf.add_to_collection('rpn_loss_box', rpn_loss_box)
                tf.add_to_collection('rpn_loss_seg', seg_loss)
                tf.add_to_collection('loss_wd', loss_wd)
                tf.add_to_collection('total_loss', loss)

            return loss

    def get_train_collection(self):
        ret = dict()
        rpn_cross_entropy_pos = tf.add_n(tf.get_collection('rpn_cross_entropy_pos')) / cfg.num_gpu
        rpn_cross_entropy_neg = tf.add_n(tf.get_collection('rpn_cross_entropy_neg')) / cfg.num_gpu
        ret['rpn_loss_box'] = tf.add_n(tf.get_collection('rpn_loss_box')) / cfg.num_gpu
        ret['rpn_loss_cat'] = tf.add_n(tf.get_collection('rpn_cross_entropy'))  / cfg.num_gpu
        ret['rpn_cross_entropy_pos'] = rpn_cross_entropy_pos
        ret['rpn_cross_entropy_neg'] = rpn_cross_entropy_neg
        ret['rpn_loss_seg'] = tf.add_n(tf.get_collection('rpn_loss_seg')) / cfg.num_gpu
        ret['loss_wd'] = tf.add_n(tf.get_collection('loss_wd')) / cfg.num_gpu
        ret['total_loss'] = tf.add_n(tf.get_collection('total_loss')) / cfg.num_gpu
        tf.summary.scalar('rpn_cross_entropy_pos', rpn_cross_entropy_pos)
        tf.summary.scalar('rpn_cross_entropy_neg', rpn_cross_entropy_neg)
        tf.summary.scalar('rpn_loss_box', ret['rpn_loss_box'])
        tf.summary.scalar('rpn_loss_cat', ret['rpn_loss_cat'])
        tf.summary.scalar('rpn_loss_seg', ret['rpn_loss_seg'])
        tf.summary.scalar('total_loss', ret['total_loss'])
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        ret['summaries'] = tf.summary.merge(summaries)
        return ret
    def get_test_collection(self):
        ret = dict()
        ret['rpn_cls_prob'] = tf.get_collection('rpn_cls_prob')[0]
        ret['rpn_bbox_pred'] = tf.get_collection('rpn_bbox_pred')[0]
        ret['anchors'] = tf.get_collection('anchors')[0]
        ret['seg_pred_pyramid'] = tf.get_collection('seg_pred_pyramid')[0]
        return ret
