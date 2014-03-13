#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin cifar10_18pct_solver.prototxt

#reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/train_net.bin cifar10_18pct_solver_lr1.prototxt cifar10_18pct_iter_60000.solverstate

#reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/train_net.bin cifar10_18pct_solver_lr2.prototxt cifar10_18pct_iter_65000.solverstate
