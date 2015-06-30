#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/triplet/lfw_triplet_solver.prototxt
