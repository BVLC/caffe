#!/usr/bin/env sh

# Check if TOOLS is unset
if [ -z "$TOOLS" ]; then
TOOLS=./build/tools
fi

$TOOLS/caffe train \
    --solver=models/bvlc_reference_caffenet/solver.prototxt
