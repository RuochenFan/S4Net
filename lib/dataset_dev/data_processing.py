# -*- coding: utf-8 -*-
import tensorflow as tf


#########for detection gt boxes##############

# flip horizintal

# todo if there need to delete 1 pixel???
def flip_gt_boxes(gt_boxes, ih, iw):
    x1s, y1s, x2s, y2s, cls = \
        gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], \
        gt_boxes[:, 3], gt_boxes[:, 4]
    x1s = tf.to_float(iw - 1) - x1s
    x2s = tf.to_float(iw - 1) - x2s
    return tf.concat(values=(x2s[:, tf.newaxis],
                             y1s[:, tf.newaxis],
                             x1s[:, tf.newaxis],
                             y2s[:, tf.newaxis],
                             cls[:, tf.newaxis]), axis=1)


def resize_gt_boxes(gt_boxes, scale_ratio):
    xys, cls = gt_boxes[:, 0:4], gt_boxes[:, 4]
    xys = xys * scale_ratio
    return tf.concat(values=(xys, cls[:, tf.newaxis]), axis=1)


def flip_image(image):
    return tf.reverse(image, axis=[1])


def smallest_size_at_least(height, width, min_side, max_side):
    min_side = tf.convert_to_tensor(min_side, dtype=tf.int32)
    min_side = tf.to_float(min_side)

    max_side = tf.convert_to_tensor(max_side, dtype=tf.int32)
    max_side = tf.to_int32(max_side)

    height = tf.to_float(height)
    width = tf.to_float(width)

    im_size_min = tf.minimum(height, width)
    im_size_max = tf.maximum(height, width)
    im_scale = min_side / im_size_min

    tf.cond(tf.greater(tf.to_int32(im_scale * im_size_max), max_side),
            lambda: tf.to_float(max_side) / im_size_max,
            lambda: im_scale)

    # scale = tf.cond(tf.greater(height, width),
    #                lambda: min_side / width,
    #                lambda: min_side / height)
    new_height = tf.to_int32(height * im_scale)
    new_width = tf.to_int32(width * im_scale)
    # new_height = tf.minimum(tf.to_int32(height * scale), max_side)
    # new_width = tf.minimum(tf.to_int32(width * scale), max_side)
    return new_height, new_width, im_scale


def sub_mean_pixel(image, means):
    num_channels = image.get_shape().as_list()[-1]
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)
