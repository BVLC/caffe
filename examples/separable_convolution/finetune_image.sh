#!/usr/bin/env sh

TOOLS=../../build/tools
MODEL=../imagenet/caffe_reference_imagenet_model

GLOG_logtostderr=1 $TOOLS/caffe train -weights=$MODEL --solver=imagenet_solver.prototxt

echo "Done."
