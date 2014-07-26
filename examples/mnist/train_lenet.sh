#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe.bin train \
  --solver_proto_file=lenet_solver.prototxt
