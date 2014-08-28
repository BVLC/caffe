#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/imagenet/alexnet_solver.prototxt

echo "Done."
