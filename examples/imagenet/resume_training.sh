#!/usr/bin/env sh

TOOLS=../../build/tools

$TOOLS/caffe train\
    --solver=imagenet_solver.prototxt \
    --snapshot=caffe_imagenet_train_10000.solverstate

echo "Done."
