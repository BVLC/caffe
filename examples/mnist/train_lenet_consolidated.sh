#!/usr/bin/env sh

# Check if TOOLS is set
if [ -z "$TOOLS" ]; then
  TOOLS=./build/tools
fi

$TOOLS/caffe train \
  --solver=examples/mnist/lenet_consolidated_solver.prototxt
