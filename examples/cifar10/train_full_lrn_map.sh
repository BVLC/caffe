#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    cifar10_full_lrn_map_solver.prototxt

#reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/train_net.bin \
    cifar10_full_lrn_map_solver_lr1.prototxt \
    cifar10_full_lrn_map_iter_60000.solverstate

#reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/train_net.bin \
    cifar10_full_lrn_map_solver_lr2.prototxt \
    cifar10_full_lrn_map_iter_65000.solverstate
