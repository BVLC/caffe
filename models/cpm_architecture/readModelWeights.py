# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:58:57 2016

@author: denitome
"""

import caffe
import os

caffe.set_mode_cpu()
caffe.set_device(1)
home_dir = os.path.expanduser('~')
def_file = '%s/MATLAB/convolutional-pose-machines-release/model/_trained_MPI/pose_deploy_centerMap.prototxt' % home_dir
model_file = '%s/MATLAB/convolutional-pose-machines-release/model/_trained_MPI/pose_iter_985000_addLEEDS.caffemodel' % home_dir
if ((not os.path.isfile(def_file)) or (not os.path.isfile(model_file))):
    print 'ERROR'

net1 = caffe.Net(def_file, model_file , caffe.TEST)
layer_names1 = list(net1._layer_names)

def_file = '%s/Libraries/caffe_cpm/models/cpm_architecture/prototxt/pose_deploy.prototxt' % home_dir
if (not os.path.isfile(def_file)):
    print 'ERROR' 

net2 = caffe.Net(def_file, caffe.TEST)
layer_names2 = list(net2._layer_names)

print '\n\nORIGINAL NETWORK\n'
for i in range(1,len(layer_names1)):
    name = layer_names1[i]
    if not('conv' in name):
        continue
    layers = net1.params[layer_names1[i]][0];
    print 'layer %s -> size: (%d, %d, %d)' % (name, layers.width, layers.height, layers.channels)

print '\n\nNEW NETWORK\n'
for i in range(1,len(layer_names2)):
    name = layer_names2[i]
    if not('conv' in name):
        continue
    layers = net2.params[layer_names2[i]][0];
    print 'layer %s -> size: (%d, %d, %d)' % (name, layers.width, layers.height, layers.channels)
    
print '\n\nBOTH\n'
for i in range(1,len(layer_names2)):
    name = layer_names2[i]
    if not('conv' in name):
        continue
    layers1 = net1.params[layer_names1[i]][0];
    layers2 = net2.params[layer_names2[i]][0];
    print 'layer %s -> size: (%d, %d, %d, %d) -> (%d, %d, %d, %d)' % (name, layers1.width, layers1.height, layers1.channels, layers1.num, layers2.width, layers2.height, layers2.channels, layers2.num)