#!/usr/bin/env sh

TOOLS=../../build/tools

export LD_LIBRARY_PATH=/usr/local/cuda-5.5/lib64:$LD_LIBRARY_PATH

GLOG_alsologtostderr=1 GLOG_stderrthreshold=0 $TOOLS/train_net.bin \
    imagenet-clarifai_solver.prototxt 2>&1 | tee ./log/log2.txt
# GLOG_logtostderr=1 $TOOLS/train_net.bin imagenet_solver.prototxt

echo "Done."
