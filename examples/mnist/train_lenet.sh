#!/usr/bin/env sh

TOOLS=../../build/tools

$TOOLS/caffe train --solver=lenet_solver.prototxt
