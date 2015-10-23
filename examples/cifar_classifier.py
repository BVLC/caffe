__author__ = 'pittnuts'

'''
Main script to run classification/test/prediction/evaluation cifar
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import *
from PIL import Image
import caffe
import sys
import lmdb
from caffe.proto import caffe_pb2
from pittnuts import *
from os import system
from caffe_apps import *

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
caffe_root = './'

imagenet_val_path  = 'examples/cifar10/cifar10_test_lmdb/'

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

# [10000] Accuracy (Top 1): 81.45%  (Top 5): 99.21% cifar10_full_scnn.prototxt before sparsifying
net = caffe.Net(caffe_root + 'examples/cifar10/cifar10_full.prototxt',
                caffe_root + 'examples/cifar10/cifar10_full_scnn_iter_70000_zerout.caffemodel',
                #caffe_root + 'examples/cifar10/eilab_cifar10_full_ini_sparsenet.caffemodel',
                caffe.TEST)

# set net to batch size
height = 32
width = 32
if height!=width:
    warnings.warn("height!=width, please double check their dimension position",RuntimeWarning)

net.blobs['data'].reshape(1,3,height,width)

#im = np.array(Image.open(caffe_root+'examples/images/cat.jpg'))
#plt.title("original image")
#plt.imshow(im)
#plt.show()
#plt.axis('off')



#######################
count = 0
correct_top1 = 0
correct_top5 = 0
labels_set = set()
lmdb_env = lmdb.open(imagenet_val_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
#pixel_mean = np.load(caffe_root + 'examples/cifar10/mean.binaryproto').mean(1).mean(1)
meanproto = 'examples/cifar10/mean.binaryproto'
f = open(meanproto,'rb')
meanblob = caffe_pb2.BlobProto()
meanblob.ParseFromString(f.read())
f.close()
pixel_mean = array(caffe.io.blobproto_to_array(meanblob))
#pixel_mean = tile(pixel_mean.reshape([1,3]),(height*width,1)).reshape(height,width,3).transpose(2,0,1)

for key, value in lmdb_cursor:
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    label = int(datum.label)
    image = caffe.io.datum_to_array(datum)
    image = image.astype(np.uint8)


    #crop_range = range(14,14+227)
    #image = image[:,14:14+227,14:14+227]
    net.blobs['data'].data[...] = image-pixel_mean[0] #transformer.preprocess('data', image)
    out = net.forward()
    plabel = int(out['prob'][0].argmax(axis=0))
    plabel_top5 = argsort(out['prob'][0])[-1:-6:-1]
    assert plabel==plabel_top5[0]
    count = count + 1

    iscorrect = label == plabel
    correct_top1 = correct_top1 + (1 if iscorrect else 0)

    iscorrect_top5 = contains(plabel_top5,label)
    correct_top5 = correct_top5 + (1 if iscorrect_top5 else 0)

    labels_set.update([label, plabel])


    sys.stdout.write("\r[{}] Accuracy (Top 1): {:.2f}%".format(count,100.*correct_top1/count))
    sys.stdout.write("  (Top 5): %.2f%%" % (100.*correct_top5/count))
    sys.stdout.flush()

plt.show()