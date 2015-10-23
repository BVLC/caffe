__author__ = 'pittnuts'
'''
Main script to run classification/test/prediction/evaluation
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

imagenet_val_path  = '/home/wew57/cuda-workspace/SCNN_MAKEFILE_PRJ/caffe/examples/imagenet/ilsvrc12_val_lmdb'
#imagenet_val_path  = '/home/student/Public/ImageNet/ilsvrc12_val_lmdb'


import os
#if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
#    print("Downloading pre-trained CaffeNet model...")
#    #!../scripts/download_model_binary.py ../models/bvlc_reference_caffenet
#    os.system("./scripts/download_model_binary.py ./models/bvlc_reference_caffenet")

#GPU mode
caffe.set_device(0)
caffe.set_mode_gpu()

#caffe.set_mode_cpu()

#net = caffe.Net(caffe_root + 'models/eilab_reference_sparsenet/deploy_scnn.prototxt',
#                caffe_root + 'models/eilab_reference_sparsenet/eilab_reference_sparsenet.caffemodel',
#                caffe.TEST)

net = caffe.Net(caffe_root + 'models/eilab_reference_sparsenet/deploy_scnn.prototxt',
                'models/eilab_reference_sparsenet/sparsenet_train_iter_160000.caffemodel',
                caffe.TEST)

#net = caffe.Net(caffe_root + 'models/eilab_reference_sparsenet/deploy_scnn_pruned.prototxt',
#                'models/eilab_reference_sparsenet/scnn_pruned.caffemodel',
#                caffe.TEST)

#net = caffe.Net(caffe_root + 'examples/cifar10/cifar10_quick_scnn.prototxt',
#                caffe_root + 'examples/cifar10/cifar10_quick_iter_5000_sparsenet.caffemodel',
#                caffe.TEST)

#net = caffe.Net(caffe_root + 'models/eilab_reference_sparsenet/deploy_scnn.prototxt',
#                caffe_root + 'models/eilab_reference_sparsenet/sparsenet_train_iter_10000_l1.caffemodel',
#                caffe.TEST)

#net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
#                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
#                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
#transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size
height = 227
width = 227
if height!=width:
    warnings.warn("height!=width, please double check their dimension position",RuntimeWarning)

net.blobs['data'].reshape(1,3,height,width)

#im = np.array(Image.open(caffe_root+'examples/images/cat.jpg'))
#plt.title("original image")
#plt.imshow(im)
#plt.show()
#plt.axis('off')

'''
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'examples/images/cat.jpg'))
out = net.forward()
print("Predicted class is #{}.".format(out['prob'].argmax()))

# load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
try:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
    print "Downloading ilsvrc12 synset"
    os.system("../data/ilsvrc12/get_ilsvrc_aux.sh")
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

# sort top k predictions from softmax output
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print labels[top_k]

# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))

feat = net.blobs['conv1'].data[0, :36]
vis_square(feat, padval=1)
'''
########## PCA #############
#for idx in range(0,5):
#    data = net.params[net.params.keys()[idx]][0].data
#    #if idx != 0:
#    #    data = data[0:data.shape[0]/2,:,:,:]
#    weights = transpose(data,(2,3,1,0))
#    P,S,Q,q = kernel_factorization(weights)
#    print weights.shape
#    #print q
#    weights_recover,R = kernel_recover(P,S,Q,q)
#    print "[{}] W: %{} error".format(net.params.keys()[idx],100*sum((weights-weights_recover)**2)/sum((weights)**2))
#    r_width = 0.0001
#    print "[{}] S: %{} zeros".format(net.params.keys()[idx],100*sum((abs(S)<r_width).flatten())/(float)(S.size))
#    print "[{}] R: %{} zeros".format(net.params.keys()[idx],100*sum((abs(R)<r_width).flatten())/(float)(R.size))
#    net.params[net.params.keys()[idx]][0].data[:] = weights_recover.transpose((3,2,0,1))

#######################
count = 0
correct_top1 = 0
correct_top5 = 0
labels_set = set()
lmdb_env = lmdb.open(imagenet_val_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
pixel_mean = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)
pixel_mean = tile(pixel_mean.reshape([1,3]),(height*width,1)).reshape(height,width,3).transpose(2,0,1)

for key, value in lmdb_cursor:
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    label = int(datum.label)
    image = caffe.io.datum_to_array(datum)
    #net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'examples/images/cat.jpg'))
    image = image.astype(np.uint8)

    #out = net.forward_all(data=np.asarray([image]))
    #image_tmp = image[(0,1,2),:,:]
    #image_tmp = image_tmp.transpose(0,2,1)
    #plt.imshow(image.transpose(1,2,0)[:,:,(2,1,0)])
    #plt.show()
    crop_range = range(14,14+227)
    image = image[:,14:14+227,14:14+227]
    net.blobs['data'].data[...] = image-pixel_mean #transformer.preprocess('data', image)
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

    #if not iscorrect:
    #print("\rError: key=%s, expected %i but predicted %i" % (key, label, plabel))

    #sys.stdout.write("\rAccuracy (Top 1): %.1f%%" % (100.*correct_top1/count))
    sys.stdout.write("\n[{}] Accuracy (Top 1): {:.1f}%".format(count,100.*correct_top1/count))
    sys.stdout.write("  (Top 5): %.1f%%" % (100.*correct_top5/count))
    sys.stdout.flush()

#print(str(correct_top1) + " out of " + str(count) + " were classified correctly")

plt.show()