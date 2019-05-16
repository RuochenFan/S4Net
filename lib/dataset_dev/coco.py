# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

from IPython import embed
import os.path as osp
import os, math, sys, glob
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
from PIL import Image
from config import cfg
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask
from dataset_dev import data_processing
import numpy.random as npr

from detection_opr.utils import vis_det
import cv2


class coco:
    def __init__(self, image_set, year):
        self._year = str(year)
        self._image_set = image_set
        self._data_path = osp.join(cfg.DATA_DIR, 'coco')
        self.record_dir = osp.join(self._data_path, 'tf_record', image_set)

        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)
            self._add_to_tfrecord(image_set + self._year)

    def read(self, is_train=True, rnd_flip=True, batch=1):
        file_pattern = '/coco_' + self._image_set + self._year + '*.tfrecord'
        records_filename = glob.glob(self.record_dir + file_pattern)
        if not isinstance(records_filename, list):
            records_filename = [records_filename]

        filename_queue = tf.train.string_input_producer(
            records_filename, num_epochs=100)
        options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
        reader = tf.TFRecordReader(options=options)
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image/img_id': tf.FixedLenFeature([], tf.int64),
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'label/num_instances': tf.FixedLenFeature([], tf.int64),
                'label/gt_boxes': tf.FixedLenFeature([], tf.string),
            })
        img_id = tf.cast(features['image/img_id'], tf.int32)
        ih = tf.cast(features['image/height'], tf.int32)
        iw = tf.cast(features['image/width'], tf.int32)
        num_instances = tf.cast(features['label/num_instances'], tf.int32)
        image = tf.decode_raw(features['image/encoded'], tf.uint8)
        im_size = tf.size(image)

        image = tf.cond(tf.equal(im_size, ih * iw),
                        lambda: tf.image.grayscale_to_rgb(
                            tf.reshape(image, (ih, iw, 1))),
                        lambda: tf.reshape(image, (ih, iw, 3)))

        gt_boxes = tf.decode_raw(features['label/gt_boxes'], tf.float32)
        gt_boxes = tf.reshape(gt_boxes, [num_instances, 5])

        if is_train:
            image, gt_boxes, ih, iw, scale = self.preprocess_for_training(
                image, gt_boxes, ih, iw, rnd_flip)
        im_imfo = tf.convert_to_tensor([[tf.to_float(ih),
                                         tf.to_float(iw), tf.to_float(scale)]])
        return image, im_imfo, gt_boxes, num_instances, img_id

    def _get_ann_file(self):
        prefix = 'instances' if self._image_set.find('test') == -1 \
            else 'image_info'
        return osp.join(self._data_path, 'annotations', prefix + '_' + \
                        self._image_set + str(self._year) + '.json')

    def _add_to_tfrecord(self, split_name):
        image_dir = self._data_path
        assert split_name in ['train2014', 'val2014', 'valminusminival2014',
                              'minival2014', 'trainmini2014']
        coco = COCO(self._get_ann_file())
        # cats = coco.loadCats(coco.getCatIds())
        print('%s has %d images' % (split_name, len(coco.imgs)))
        imgs = [(img_id, coco.imgs[img_id]) for img_id in coco.imgs]
        num_shards = int(len(imgs) / 2500)
        num_per_shard = int(math.ceil(len(imgs) / float(num_shards)))
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            sess = tf.Session('')
            for shard_id in range(num_shards):
                record_filename = self._get_dataset_filename(
                    split_name, shard_id, num_shards)
                options = tf.python_io.TFRecordOptions(
                    TFRecordCompressionType.ZLIB)
                tf_record_writer = tf.python_io.TFRecordWriter(
                    record_filename, options=options)
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id + 1) * num_per_shard, len(imgs))
                for i in range(start_ndx, end_ndx):
                    if i % 1 == 0:
                        sys.stdout.write(
                            '\r>> Converting image %d/%d shard %d\n' % (
                                i + 1, len(imgs), shard_id))
                        sys.stdout.flush()

                    img_id = imgs[i][0]
                    img_name = imgs[i][1]['file_name']
                    split = img_name.split('_')[1]
                    img_name = os.path.join(image_dir, split, img_name)
                    if str(img_id) == '320612' or img_id == 167126:
                        continue
                    height, width = imgs[i][1]['height'], imgs[i][1]['width']

                    gt_boxes = self._get_coco_gt(
                        coco, img_id, height, width, img_name)

                    # be careful, Image.open is different from cv2.imread
                    img = np.array(Image.open(img_name))
                    if img.size == height * width:
                        print('Gray Image %s' % str(img_id))
                        im = np.empty((height, width, 3), dtype=np.uint8)
                        im[:, :, :] = img[:, :, np.newaxis]
                        img = im

                    img = img.astype(np.uint8)
                    assert img.size == width * height * 3, '%s' % str(img_id)

                    # debug vis
                    # tmp_gt_boxes = []
                    # for bbox in gt_boxes:
                    #     tmp_gt_boxes.append(
                    #         np.array([*bbox[0:4], 1.0, bbox[4]]))
                    # tmp_gt_boxes = np.array(tmp_gt_boxes)
                    # vis_det.visualize_detection_old(img, tmp_gt_boxes,
                    #                                 classes=class_names)

                    if len(gt_boxes) == 0:
                        continue
                    img_raw = img.tostring()
                    example = tf.train.Example(
                        features=tf.train.Features(feature={
                            'image/img_id': _int64_feature(img_id),
                            'image/encoded': _bytes_feature(img_raw),
                            'image/height': _int64_feature(height),
                            'image/width': _int64_feature(width),
                            'label/num_instances': _int64_feature(
                                gt_boxes.shape[0]),
                            'label/gt_boxes': _bytes_feature(
                                gt_boxes.tostring())
                        }))
                    tf_record_writer.write(example.SerializeToString())
                tf_record_writer.close()
            sess.close()

    def _get_dataset_filename(self, split_name, shard_id, num_shards):
        output_filename = 'coco_%s_%05d-of-%05d.tfrecord' % (
            split_name, shard_id, num_shards)
        return os.path.join(self.record_dir, output_filename)

    def _get_coco_gt(self, coco, img_id, height, width, img_name):
        annIds = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(annIds)
        bboxes = []
        classes = []
        for ann in anns:
            if ann['iscrowd']:
                continue
            cat_id = _cat_id_to_real_id(ann['category_id'])
            classes.append(cat_id)
            bboxes.append(ann['bbox'])

        bboxes = np.asarray(bboxes)
        classes = np.asarray(classes)
        if bboxes.shape[0] <= 0:
            bboxes = np.zeros([0, 4], dtype=np.float32)
            classes = np.zeros([0], dtype=np.float32)
            print ('None Annotations %s' % img_name)

        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] - 1
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] - 1
        # todo add boxes contrain, 0-width
        assert ((bboxes[:, 0] >= 0).all() and (bboxes[:, 1] >= 0).all())
        assert ((bboxes[:, 2] < width).all() and (bboxes[:, 3] < height).all())

        gt_boxes = np.hstack((bboxes, classes[:, np.newaxis]))
        gt_boxes = gt_boxes.astype(np.float32)
        return gt_boxes

    # train data processing, follow the step of py faster
    def preprocess_for_training(self, image, gt_boxes, ih, iw, rnd_flip):
        ih, iw = tf.shape(image)[0], tf.shape(image)[1]

        rnd_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES))
        new_ih, new_iw, scale_ratio = data_processing.smallest_size_at_least(
            ih, iw, cfg.TRAIN.SCALES[rnd_scale_ind], cfg.TRAIN.MAX_SIZE)

        coin = tf.to_float(tf.random_uniform([1]))[0]
        if rnd_flip:
            image, gt_boxes = tf.cond(
                tf.greater_equal(coin, 0.5),
                lambda: (data_processing.flip_image(image),
                         data_processing.flip_gt_boxes(
                             gt_boxes, ih, iw)),
                lambda: (image, gt_boxes))

        ############ reshape gtboxes ##################
        gt_boxes = data_processing.resize_gt_boxes(gt_boxes, scale_ratio)

        ############ reshape image #################
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(
            image, [new_ih, new_iw], align_corners=False)
        image = tf.squeeze(image, axis=[0])

        ###reverse image channel, for Image.open channel is dif from cv2
        image = tf.reverse(image, axis=[-1])
        image = tf.cast(image, tf.float32)
        image = data_processing.sub_mean_pixel(
            image, tf.to_float(cfg.PIXEL_MEANS[0][0]))
        image = tf.expand_dims(image, axis=0)
        return image, gt_boxes, new_ih, new_iw, scale_ratio

    ######################todo for  evaluation ##########################
    def evaluate_detections(self, all_boxes, output_dir, logger):
        pass


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _cat_id_to_real_id(readId):
    """Note coco has 80 classes, but the catId ranges from 1 to 90!"""
    cat_id_to_real_id = \
        {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
         13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
         18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25,
         28: 26, 31: 27, 32: 28, 33: 29, 34: 30,
         35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39,
         44: 40, 46: 41, 47: 42, 48: 43, 49: 44,
         50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53,
         59: 54, 60: 55, 61: 56, 62: 57, 63: 58,
         64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67,
         77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
         82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
    return cat_id_to_real_id[readId]


class_names = ['background', 'person', 'bicycle', 'car', 'motorcycle',
               'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
               'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
               'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

if __name__ == '__main__':
    pass
