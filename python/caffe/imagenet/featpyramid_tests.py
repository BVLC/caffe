'''
test cases for multiscale pyramids of Convnet features.

    used power_wrapper.py as a starting point.


example usage:
python featpyramid_tests.py --images_file=image_cat.txt --crop_mode=selective_search --model_def=../../../examples/imagenet_deploy.prototxt --pretrained_model=../../../alexnet_train_iter_470000 --output=selective_cat.h5

'''

import numpy as np
import os
import sys
import gflags
import pandas as pd
import time
import skimage.io
import skimage.transform
import selective_search_ijcv_with_python as selective_search
import caffe

#parameters to consider passing to C++ Caffe::featpyramid...
# image filename
# num scales (or something to control this)
# padding amount
# [batchsize is defined in prototxt... fine.]

#hopefully caffenet is passed by ref...
def test_pyramid_IO(caffenet, imgFname):
    #TODO

    #caffenet.testIO()
    example_np_array = caffenet.testIO() #just return an array with 1 2 3 4...
    #example_np_arrays = caffenet.testIO_multiPlane() #return array of arrays

    print example_np_array
    print example_np_array[0].shape

    caffenet.testString('hi')
    caffenet.testInt(1337)
'''
def featpyramid(caffenet, imgFname, [params]):
    stitched_pyra = caffenet.extract_featpyramid(imgFname, [params])    

    pyra = unstitch_pyra(stitched_pyra) #alternatively, do this in C++ inside extract_featpyramid()

    return pyra
'''


if __name__ == "__main__":

    #pretend that these flags came off the command line:
    imgFname = './pascal_009959.jpg'
    model_def = '../../../examples/imagenet_deploy.prototxt' 
    pretrained_model = '../../../alexnet_train_iter_470000'
    use_gpu = True
    
    caffenet = caffe.CaffeNet(model_def, pretrained_model)
    caffenet.set_phase_test()
    if use_gpu:
        caffenet.set_mode_gpu()


    #experiments...
    test_pyramid_IO(caffenet, imgFname)


