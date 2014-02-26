#!/usr/bin/env sh

GLOG_logtostderr=1 ../build/tools/train_net.bin lenet_solver.prototxt
