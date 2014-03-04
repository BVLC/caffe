'''
test cases for multiscale pyramids of Convnet features.

    used power_wrapper.py as a starting point.
'''

import numpy as np
import os
import sys
#import gflags
import time
import caffe
import time 

#for visualization, can be removed easily:
from matplotlib import cm, pyplot
import pylab

def test_featpyramid_allScales(caffenet, imgFname):

    densenet_params = dict()
    densenet_params['interval'] = 10
    densenet_params['img_padding'] = 16
    densenet_params['feat_minWidth'] = 10 #smallest desired scale, in terms of feature map dims
    densenet_params['feat_minHeight'] = 10

    start_time = time.time()
    pyra = caffenet.extract_featpyramid(imgFname, densenet_params) # THE CRUX ...
    # pyra = caffenet.extract_featpyramid(imgFname) # ... or THE CRUX without parameters
    pyra_time = time.time() - start_time
    print "    computed pyra in %f sec" %pyra_time

    feat = pyra["feat"]

    output_dir = 'output_pyra'
    try:
        os.makedirs( output_dir )
    except OSError, e:
        pass

    for i in xrange(0, len(feat)):
        print 'feat[%d] shape: '%i
        print feat[i].shape

        #prep for visualization (sum over depth of descriptors)
        flat_descriptor = np.sum(feat[i], axis=1) #e.g. (1, depth=256, height=124, width=124) -> (1, 124, 124) 
        flat_descriptor = flat_descriptor[0] #(1, 124, 124) -> (124, 124) ... first image (in a batch size of 1)

        #visualization
        pyplot.figure()
        pyplot.title('Welcome to deep learning land. You have arrived.')
        pyplot.imshow(flat_descriptor, cmap = cm.gray, interpolation='nearest')
        pylab.savefig( output_dir + '/flat_descriptor_scale%d.jpg' % i)

    print "\n pyra scales:"
    print pyra["scales"]
    print "pyra[imwidth] = %d, pyra[imheight]=%d" % (pyra['imwidth'], pyra['imheight'])

if __name__ == "__main__":

    imgFname = './imagenet/pascal_009959.jpg'
    model_def = './imagenet/imagenet_rcnn_batch_1_input_2000x2000_output_conv5.prototxt' 
    pretrained_model = '../../examples/alexnet_train_iter_470000'
    use_gpu = False
    #use_gpu = True
    
    caffenet = caffe.CaffeNet(model_def, pretrained_model)
    caffenet.set_phase_test()
    if use_gpu:
        caffenet.set_mode_gpu()
    else:
        caffenet.set_mode_cpu()

    test_featpyramid_allScales(caffenet, imgFname)
