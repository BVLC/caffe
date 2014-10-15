#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=./examples/siamese/ $TOOLS/caffe train --solver=examples/siamese/mnist_siamese_solver.prototxt
