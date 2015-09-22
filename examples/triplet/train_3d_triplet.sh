#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/triplet/3d_triplet_solver.prototxt
