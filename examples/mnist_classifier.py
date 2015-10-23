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

imagenet_val_path  = 'examples/mnist/mnist_test_lmdb/'

# GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net(caffe_root + 'examples/mnist/lenet_relu.prototxt',
                caffe_root + 'examples/mnist/lenet_iter_10000.caffemodel',
                #caffe_root + 'examples/cifar10/eilab_cifar10_full_ini_sparsenet.caffemodel',
                caffe.TEST)

for idx in range(0,len(net.params.keys())):
    key = net.params.keys()[idx]
    weights = net.params[key][0].data
    print '{}:{}'.format(key,weights.shape)
    if len(weights.shape)<=2:
        np.savetxt(key+".weight",weights,fmt='%.8f')
        np.savetxt(key+".bias", net.params[key][1].data,fmt='%.8f')

# set net to batch size
height = 28
width = 28
if height!=width:
    warnings.warn("height!=width, please double check their dimension position",RuntimeWarning)

net.blobs['data'].reshape(1,1,height,width)

#######################
count = 0
correct_top1 = 0
correct_top5 = 0
labels_set = set()
lmdb_env = lmdb.open(imagenet_val_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
#meanproto = 'examples/cifar10/mean.binaryproto'

#f = open(meanproto,'rb')
#meanblob = caffe_pb2.BlobProto()
#meanblob.ParseFromString(f.read())
#f.close()
#pixel_mean = array(caffe.io.blobproto_to_array(meanblob))

#layer_prop_from=('conv1','conv2','ip1','ip2')
#layer_prop_to=('pool1','pool2','relu1','prob')
layer_prop_from=('conv1','ip1','ip2')
layer_prop_to=('pool2','relu1','prob')
average_sparsity = zeros((1,len(layer_prop_from)))
for key, value in lmdb_cursor:
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    label = int(datum.label)
    image = caffe.io.datum_to_array(datum)
    image = image.astype(np.uint8)


    net.blobs['data'].data[...] = image*0.00390625#-pixel_mean[0] #transformer.preprocess('data', image)
    #out = net.forward()
    for prop_step in range(0,len(layer_prop_from)):
        end_layername = layer_prop_to[prop_step]
        #print end_layername
        out = net.forward(start=layer_prop_from[prop_step],end=end_layername)
        if prop_step==0 and count<100:
            np.savetxt(layer_prop_from[prop_step+1]+".featruemap{}".format(count),reshape(out[end_layername],(out[end_layername].size,1)), fmt='%.8f')
        #show_filter_outputs(net,end_layername)
        #plt.show()
        tmp_out = abs(out[end_layername]).flatten()
        thre = 1#tmp_out[argsort(tmp_out)[round(tmp_out.size*40/100)]]
        thre = max([0,thre])
        if prop_step!=len(layer_prop_from)-1:
            zero_out(net.blobs[end_layername].data,thre)
            average_sparsity[0,prop_step] = (average_sparsity[0,prop_step]*count + get_sparsity(net.blobs[end_layername].data,0.0001))/(count+1)

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
    sys.stdout.write("  (sparsity): " + array_str(average_sparsity))
    sys.stdout.flush()

plt.show()