__author__ = 'pittnuts'

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
import argparse
import bottleneck as bn

# helper show filter outputs
def show_filter_outputs(net,blobname):
    if len(net.blobs[blobname].data.shape) < 3:
        return
    feature_map_num = net.blobs[blobname].data.shape[1]
    plt.figure()
    filt_min, filt_max = net.blobs[blobname].data.min(), net.blobs[blobname].data.max()
    display_region_size = ceil(sqrt(feature_map_num))
    for i in range(feature_map_num):
        plt.subplot((int)(display_region_size),(int)(display_region_size),i+1)
        #plt.title("filter #{} output".format(i))
        plt.imshow(net.blobs[blobname].data[0,i], vmin=filt_min, vmax=filt_max)
        #plt.tight_layout()
        plt.axis('off')


#--imagenet_val_path examples/imagenet/ilsvrc12_val_lmdb --prototxt models/eilab_reference_sparsenet/deploy_scnn.prototxt --caffemodel models/eilab_reference_sparsenet/eilab_reference_sparsenet_zerout.caffemodel
#--imagenet_val_path examples/imagenet/ilsvrc12_val_lmdb --prototxt models/bvlc_reference_caffenet/deploy.prototxt --caffemodel models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prototxt', type=str, required=True)
    parser.add_argument('--imagenet_val_path', type=str, required=True)
    parser.add_argument('--caffemodel', type=str, required=True)
    args = parser.parse_args()
    prototxt=args.prototxt
    imagenet_val_path=args.imagenet_val_path
    caffedmodel=args.caffemodel

    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = caffe.Net( prototxt, caffedmodel, caffe.TEST)
    print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
    height = 227
    width = 227
    if height!=width:
        warnings.warn("height!=width, please double check their dimension position",RuntimeWarning)

    net.blobs['data'].reshape(1,3,height,width)
    count = 0
    correct_top1 = 0
    correct_top5 = 0
    labels_set = set()
    lmdb_env = lmdb.open(imagenet_val_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    pixel_mean = np.load( 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)
    pixel_mean = tile(pixel_mean.reshape([1,3]),(height*width,1)).reshape(height,width,3).transpose(2,0,1)
    layers = net.layers
    #layer_prop_from=('conv1','conv2','conv3','conv4','conv5')
    #layer_prop_to=('norm1','norm2','relu3','relu4','prob')
    #layer_prop_from=('conv1','conv4','conv5')
    #layer_prop_to=('relu3','relu4','prob')
    layer_prop_from=('conv1','conv2','conv3','conv4','conv5','fc6')
    layer_prop_to=('norm1','norm2','relu3','relu4','pool5','prob')
    average_sparsity = zeros((1,len(layer_prop_from)))

    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        crop_range = range(14,14+227)
        image = image[:,14:14+227,14:14+227]
        #plt.imshow(image.transpose(1,2,0))
        #plt.show()
        net.blobs['data'].data[...] = image-pixel_mean #transformer.preprocess('data', image)
        for prop_step in range(0,len(layer_prop_from)):
            end_layername = layer_prop_to[prop_step]
            #print end_layername
            out = net.forward(start=layer_prop_from[prop_step],end=end_layername)
            #show_filter_outputs(net,end_layername)
            #plt.show()
            tmp_out = abs(out[end_layername]).flatten()
            thre = 0#tmp_out[argsort(tmp_out)[round(tmp_out.size*40/100)]]
            thre = max([1,thre])
            if prop_step!=len(layer_prop_from)-1:
                zero_out(net.blobs[end_layername].data,thre)
                average_sparsity[0,prop_step] = (average_sparsity[0,prop_step]*count + get_sparsity(net.blobs[end_layername].data,0.0001))/(count+1)

            #cur_sparsity =

        #show_filter_outputs(net,'data')
        #show_filter_outputs(net,'relu3')
        #plt.show()
        plabel = int(out['prob'][0].argmax(axis=0))
        plabel_top5 = argsort(out['prob'][0])[-1:-6:-1]
        assert plabel==plabel_top5[0]
        count = count + 1

        iscorrect = label == plabel
        correct_top1 = correct_top1 + (1 if iscorrect else 0)

        iscorrect_top5 = contains(plabel_top5,label)
        correct_top5 = correct_top5 + (1 if iscorrect_top5 else 0)

        labels_set.update([label, plabel])

        sys.stdout.write("\n[{}] Accuracy (Top 1): {:.1f}%".format(count,100.*correct_top1/count))
        sys.stdout.write("  (Top 5): %.1f%%" % (100.*correct_top5/count))
        sys.stdout.write("  (sparsity): " + array_str(average_sparsity))
        sys.stdout.flush()