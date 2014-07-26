#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe.bin train \
    --solver_proto_file=cifar10_full_solver.prototxt

#reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/caffe.bin train \
    --solver_proto_file=cifar10_full_solver_lr1.prototxt \
    --resume_point_file=cifar10_full_iter_60000.solverstate

#reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/caffe.bin train \
    --solver_proto_file=cifar10_full_solver_lr2.prototxt \
    --resume_point_file=cifar10_full_iter_65000.solverstate
