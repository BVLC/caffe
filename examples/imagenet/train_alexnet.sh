#!/usr/bin/env sh

TOOLS=../../build/tools

$TOOLS/caffe train --solver=alexnet_solver.prototxt

echo "Done."
