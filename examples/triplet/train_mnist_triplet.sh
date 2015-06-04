#!/usr/bin/env sh

TOOLS=./release/tools

$TOOLS/caffe train --solver=examples/triplet/mnist_triplet_solver.prototxt
