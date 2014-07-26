#!/bin/bash
TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe.bin train \
  --solver_proto_file=mnist_autoencoder_solver.prototxt
