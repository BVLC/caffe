#!/bin/bash
TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin mnist_autoencoder_solver.prototxt
