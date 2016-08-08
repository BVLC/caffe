#!/usr/bin/env sh

# Check if CAFFE_BIN is unset
if [ -z "$CAFFE_BIN" ]; then
  TOOLS=./build/tools
else
  TOOLS=$CAFFE_BIN
fi

$TOOLS/caffe train \
    --solver=models/bvlc_reference_caffenet/solver.prototxt
