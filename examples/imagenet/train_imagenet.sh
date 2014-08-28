#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/imagenet/imagenet_solver.prototxt

echo "Done."
