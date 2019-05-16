# Authors: 644142239@qq.com (Ruochen Fan)

import cv2
import numpy as np
import os
import pickle as pkl
import random

def list_colors(im):
    colors = []

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j].sum() == 0:
                continue
            have_occurred = False
            for color in colors:
                if np.all(color == im[i, j, :]):
                    have_occurred = True
            if not have_occurred:
                colors.append(im[i,j,:])
    return colors

def get_bbox(im):
    im_x = np.sum(im, axis=0)
    im_y = np.sum(im, axis=1)
    x1, y1, x2, y2 = 0, 0, len(im_x)-1, len(im_y)-1
    for i in range(len(im_x)):
        if im_x[i] == 0:
            x1 += 1
        else:
            break
    for i in range(len(im_x)-1, 0, -1):
        if im_x[i] == 0:
            x2 -= 1
        else:
            break
    for i in range(len(im_y)):
        if im_y[i] == 0:
            y1 += 1
        else:
            break
    for i in range(len(im_y) - 1, 0, -1):
        if im_y[i] == 0:
            y2 -= 1
        else:
            break

    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h


dataset_dir = '../data/instance_saliency/'

files = os.listdir(dataset_dir+'gt')
random.shuffle(files)
total_samples = 1000
num_train = 700
num_test = 300

im_gts = []
gt_boxes = []
gt_segs = []
im_images = []


for file in files:
    if file.startswith('._'):
        print('----------------------------')
        continue
    print(file)
    fgt = dataset_dir+'gt/' + file
    fimage = dataset_dir+'image/' + file[:-3] + 'jpg'
    im_gt = cv2.imread(fgt)
    im_image = cv2.imread(fimage)
    im_gts.append(im_gt)
    im_images.append(im_image)

    colors = list_colors(im_gt)
    boxes = np.zeros([len(colors), 4])
    segs = np.zeros([len(colors), im_gt.shape[0], im_gt.shape[1]])
    for i, color in enumerate(colors):
        segs[i] = np.all(im_gt == color, axis=2)
        x, y, w, h = get_bbox(segs[i])
        boxes[i, 0] = x
        boxes[i, 1] = y
        boxes[i, 2] = w
        boxes[i, 3] = h

    gt_boxes.append(boxes)
    gt_segs.append(segs)

dataset = {'train_imgs':None, 'test_imgs':None, 'train_boxes':None,
           'test_boxes':None, 'train_segs':None, 'test_segs':None}
dataset['train_imgs'] = im_images[:num_train]
dataset['test_imgs'] = im_images[num_train:]
dataset['train_boxes'] = gt_boxes[:num_train]
dataset['test_boxes'] = gt_boxes[num_train:]
dataset['train_segs'] = gt_segs[:num_train]
dataset['test_segs'] = gt_segs[num_train:]

with open('../data/dataset.pkl', 'wb') as f:
    pkl.dump(dataset, f)

print('done')


