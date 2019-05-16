# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

if __name__ == '__main__':
  pass


import xml.etree.ElementTree as ET
import numpy as np

class pascal_voc():
  def __init__(self, image_set, year, devkit_path=None):
    self._year = year
    self._image_set = image_set
    self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
    self._classes = ('__background__',  # always index 0
                     'aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor')
    self._class_to_ind = dict(
      list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.jpg'
    self._image_index = self._load_image_set_index()
    self._comp_id = 'comp4'
    self.config = {'cleanup': True,
                   'use_diff': False,
                   'matlab_eval': False}
    assert os.path.exists(self._devkit_path), \
      'VOCdevkit path does not exist: {}'.format(self._devkit_path)

    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  """
  Return the absolute path to image i in the image sequence.
  """

  def image_path_at(self, i):
    return self.image_path_from_index(self._image_index[i])

  """
  Construct an image path from the image's "index" identifier.
  """

  def image_path_from_index(self, index):
    image_path = os.path.join(self._data_path, 'JPEGImages',
                              index + self._image_ext)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  """
  Load the indexes listed in this dataset's image set file.
  """

  def _load_image_set_index(self):
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                  self._image_set + '.txt')
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]
    return image_index

  def _to_tfrecord(tfrecord_dir):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
      with tf.Session('') as sess:
        for i in range(len(imgs)):
          index = 0
          filename = os.path.join(self._data_path, 'Annotations',
                                  index + '.xml')
          tree = ET.parse(filename)
          objs = tree.findall('object')
          if not self.config['use_diff']:
            non_diff_objs = [
              obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
          num_objs = len(objs)
          boxes = np.zeros((num_objs, 4), dtype=np.uint16)
          gt_classes = np.zeros((num_objs), dtype=np.int32)
          overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
          seg_areas = np.zeros((num_objs), dtype=np.float32)

          for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)



