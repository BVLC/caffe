#!/usr/bin/env sh

TOOLS=../../build/tools
MODEL=../imagenet/caffe_reference_imagenet_model

GLOG_logtostderr=1 $TOOLS/finetune_net.bin voc2012_finetune_spatial_pyramid_pooling_solver.prototxt $MODEL

echo "Done."
