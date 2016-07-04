#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLE=examples/cifar10
DATA=data/cifar10
DBTYPE=lmdb

# Check if CAFFE_BIN is unset
if [ -z "$CAFFE_BIN" ]; then
  EXAMPLES=./build/$EXAMPLE
  TOOLS=./build/tools
else
  EXAMPLES=$CAFFE_BIN
  TOOLS=$CAFFE_BIN
fi  

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/cifar10_test_$DBTYPE

$EXAMPLES/convert_cifar_data.bin $DATA $EXAMPLE $DBTYPE

echo "Computing image mean..."

$TOOLS/compute_image_mean -backend=$DBTYPE \
  $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/mean.binaryproto

echo "Done."
