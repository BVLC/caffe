#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe.bin train\
    --solver_proto_file=imagenet_solver.prototxt \
    --resume_point_file=caffe_imagenet_train_10000.solverstate

echo "Done."
