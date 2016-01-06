#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/media/dd2/projects/deepdetect/datasets/imagenet/ilsvrc15/
DATA=/media/dd2/projects/deepdetect/datasets/imagenet/ilsvrc15/
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/ilsvrc15_train_lmdb \
  $DATA/imagenet_mean.binaryproto

echo "Done."
