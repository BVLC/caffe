#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

./build/tools/compute_image_mean examples/imagenet/ilsvrc12_train_lmdb \
  data/ilsvrc12/imagenet_mean.binaryproto

echo "Done."
