#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/ilsvrc12

$TOOLS/compute_image_mean.bin ilsvrc12_train_leveldb $DATA/imagenet_mean.binaryproto

echo "Done."
