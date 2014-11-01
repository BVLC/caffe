#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLE=examples/cifar10
DATA=data/cifar10
BUILD=build/examples/cifar10
TOOLS=build/tools

# At this time, only leveldb is available for CIFAR-10
DBTYPE=leveldb

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/cifar10_train_$DBTYPE 
rm -rf $EXAMPLE/cifar10_test_$DBTYPE

$BUILD/convert_cifar_data.bin $DATA $EXAMPLE $DTYPE

echo "Computing image mean..."

$TOOLS/compute_image_mean.bin  $EXAMPLE/cifar10_train_$DBTYPE \
  $EXAMPLE/mean.binaryproto $DBTYPE

echo "Done."
