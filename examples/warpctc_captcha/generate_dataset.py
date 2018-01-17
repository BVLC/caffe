#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
from multiprocessing import Process
import caffe
import h5py

CAFFE_ROOT = os.getcwd()   # assume you are in $CAFFE_ROOT$ dir
img_path = os.path.join(CAFFE_ROOT, 'data/captcha/')
#IMAGE_WIDTH, IMAGE_HEIGHT = 80, 30
IMAGE_WIDTH, IMAGE_HEIGHT = 128, 32
LABEL_SEQ_LEN = 5
# captcha images list
images = filter(lambda x: os.path.splitext(x)[1] == '.png', os.listdir(img_path))

print '[+] total image number: {}'.format(len(images))

np.random.shuffle(images)

def write_image_info_into_file(file_name, images):
    with open(file_name, 'w') as f:
        for image in images:
            img_name = os.path.splitext(image)[0]
            numbers = img_name[img_name.find('-')+1:]
            f.write(os.path.join(img_path, image) + "|" + ','.join(numbers) + "\n")


def write_image_info_into_hdf5(file_name, images, phase):
    total_size = len(images)
    print '[+] total image for {0} is {1}'.format(file_name, len(images))
    single_size = 20000
    groups = total_size / single_size
    if total_size % single_size:
        groups += 1
    def process(file_name, images):
        img_data = np.zeros((len(images), 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype = np.float32)
        label_seq = 10*np.ones((len(images), LABEL_SEQ_LEN), dtype = np.float32)
        for i, image in enumerate(images):
            img_name = os.path.splitext(image)[0]
            numbers_str = img_name[img_name.find('-')+1:]
            numbers = np.array(map(lambda x: float(x), numbers_str))
            label_seq[i, :len(numbers)] = numbers
            img = caffe.io.load_image(os.path.join(img_path, image))
            img = caffe.io.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
            img = np.transpose(img, (2, 0, 1))
            img_data[i] = img
            """
            if (i+1) % 100 == 0:
                print '[+] name: {}'.format(image)
                print '[+] number: {}'.format(','.join(map(lambda x: str(x), numbers)))
                print '[+] label: {}'.format(','.join(map(lambda x: str(x), label_seq[i])))
            """
        with h5py.File(file_name, 'w') as f:
            f.create_dataset('data', data = img_data)
            f.create_dataset('label', data = label_seq)
    with open(file_name, 'w') as f:
        workspace = os.path.split(file_name)[0]
        process_pool = []
        for g in xrange(groups):
            h5_file_name = os.path.join(workspace, '%s_%d.h5' %(phase, g))
            f.write(h5_file_name + '\n')
            start_idx = g*single_size
            end_idx = start_idx + single_size
            if g == groups - 1:
                end_idx = len(images)
            p = Process(target = process, args = (h5_file_name, images[start_idx:end_idx]))
            p.start()
            process_pool.append(p)
        for p in process_pool:
            p.join()
trainning_size = 80000   # number of images for trainning
trainning_images = images[:trainning_size]

testing_images = images[trainning_size:]
write_image_info_into_hdf5(os.path.join(img_path, 'trainning.list'), trainning_images, 'train')
write_image_info_into_hdf5(os.path.join(img_path, 'testing.list'), testing_images, 'test')
write_image_info_into_file(os.path.join(img_path, 'testing-images.list'), testing_images)
