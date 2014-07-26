#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe.bin train \
  --solver_proto_file=cifar10_quick_solver.prototxt

#reduce learning rate by fctor of 10 after 8 epochs
GLOG_logtostderr=1 $TOOLS/caffe.bin train \
  --solver_proto_file=cifar10_quick_solver_lr1.prototxt \
  --resume_point_file=cifar10_quick_iter_4000.solverstate
