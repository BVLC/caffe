#!/bin/bash

CAFFE_ROOT=/home/jaehyun/github/caffe-fast-rcnn

TRAIN_MODEL=models/jhlim_inception_bn/train_val.prototxt
TEST_MODEL=models/jhlim_inception_bn/train_val.prototxt
WEIGHT=models/jhlim_inception_bn/inception_bn_stepsize_6400_iter_1200000.caffemodel

$CAFFE_ROOT/build/tools/test_bn -train_model $TRAIN_MODEL -test_model $TEST_MODEL -weights $WEIGHT -train_iterations 1000 -gpu 0 
