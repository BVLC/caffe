#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_reference_caffenet/solver.prototxt \
    --snapshot=models/bvlc_reference_caffenet/caffenet_train_10000.solverstate.h5
