# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""

import tensorflow as tf
import numpy as np

'''
crop_pool_layer:
    bottom: image
    stride: single number
'''


def crop_pool_layer(bottom, pool_size, stride, rois, name):
    with tf.variable_scope(name) as scope:
        batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1],
                                        name="batch_id"), [1])
        # Get the normalized coordinates of bboxes
        bottom_shape = tf.shape(bottom)
        height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(stride)
        width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(stride)
        x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
        y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
        x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
        y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
        # Won't be backpropagated to rois anyway, but to save time
        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))

        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids),
                                         pool_size, name="crops")
    return crops


# todo : optimize complex reshape layer
# current batch size is 1
def reshape_layer(bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
        to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
        reshaped = tf.reshape(
            to_caffe, tf.concat(axis=0, values=[
                [1], [num_dim, -1], [input_shape[2]]]))
        to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
    return to_tf


if __name__ == '__main__':
    pass
