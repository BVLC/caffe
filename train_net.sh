#!/bin/bash

GLOG_logtostderr=1 build/tools/caffe train \
-solver models/cpm_architecture/prototxt/pose_solver.prototxt \
-weights models/cpm_architecture/savedmodels/pose_iter_985000_addLEEDS.caffemodel \
-gpu 0 2>&1 | tee models/cpm_architecture/prototxt/log.txt