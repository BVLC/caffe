#!/usr/bin/env sh

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin lenet_solver.prototxt
