#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/mnist/lenet_local_solver.prototxt --gpu=1
