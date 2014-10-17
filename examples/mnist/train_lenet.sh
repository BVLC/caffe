#!/usr/bin/env sh

GLOG_logtostderr=0 GLOG_log_dir=examples/mnist/ ./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt
