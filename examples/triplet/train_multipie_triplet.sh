#!/usr/bin/env sh
# This script training in MULTIPIE database which takes 1 positive sample and 3
# negative samples as training data set, the negative samples are ones which are
# different from reference sample.

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/triplet/multipie_triplet_solver.prototxt