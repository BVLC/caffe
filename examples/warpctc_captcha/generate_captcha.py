#!/usr/bin/env python
# coding=utf-8
from captcha.image import ImageCaptcha
import os
import numpy as np
from multiprocessing import Process

DATASET_SIZE = 100000
LABEL_SEQ_LENGTH = 5
BLANK_LABEL = 10

def generate_random_label(length):
    """
    generate labels, we use 10 as blank
    """
    not_blank = []
    while len(not_blank) == 0:
        rand_array = np.random.randint(11, size = length)
        not_blank = rand_array[rand_array != BLANK_LABEL]

    return ''.join(map(lambda x: str(x), not_blank))

image = ImageCaptcha()
CAFFE_ROOT = os.getcwd()   # assume you are in $CAFFE_ROOT$ dir
img_path = os.path.join(CAFFE_ROOT, 'data/captcha/')

def generate_image(seed, start, end):
    np.random.seed(seed)
    for idx in xrange(start, end):
        label_seq = generate_random_label(LABEL_SEQ_LENGTH)
        image.write(label_seq, os.path.join(img_path, '%05d-'%idx + label_seq + '.png'))

threads_num = 10
threads = []
batch_size = DATASET_SIZE / threads_num

for t in xrange(threads_num):
    start, end = t*batch_size, (t+1)*batch_size
    if t == threads_num - 1:
        end = DATASET_SIZE
    p = Process(target = generate_image, args = (t, start, end))
    p.start()
    threads.append(p)
for p in threads:
    p.join()
