#!/usr/bin/env sh

TOOLS=../../build/tools
MODEL=voc2012_finetune_imagenet_spatial_pyramid_pooling_train_iter_6000

GLOG_logtostderr=1 $TOOLS/finetune_net.bin spp_net_region_classifier_solver.prototxt $MODEL

echo "Done."
