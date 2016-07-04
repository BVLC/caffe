#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/imagenet
DATA=data/ilsvrc12

# Check if CAFFE_BIN is unset
if [ -z "$CAFFE_BIN" ]; then
  TOOLS=./build/tools
else
  TOOLS=$CAFFE_BIN
fi


$TOOLS/compute_image_mean $EXAMPLE/ilsvrc12_train_lmdb \
  $DATA/imagenet_mean.binaryproto

echo "Done."
