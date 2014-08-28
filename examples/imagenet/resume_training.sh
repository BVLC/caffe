#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/imagenet/imagenet_solver.prototxt \
    --snapshot=examples/imagenet/caffe_imagenet_10000.solverstate

echo "Done."
