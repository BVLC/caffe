#!/bin/bash

GLOG_logtostderr=1 build/tools/caffe train \
-solver models/cpm_architecture/prototxt/H36M_validation/pose_solver.prototxt \
-weights models/cpm_architecture/prototxt/H36M_validation/savedmodels/pose_iter_985000_addLEEDS.caffemodel \
-gpu 0 2>&1 | tee models/cpm_architecture/log.txt