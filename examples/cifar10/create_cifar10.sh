#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLE=examples/cifar10
DATA=data/cifar10
DBTYPE=lmdb

# Check if CAFFE_ROOT is set
if [ -z ${CAFFE_ROOT+x} ]; 
# if unset
then
  EXAMPLES_BIN=./build/$EXAMPLE
  TOOLS_BIN=./build/tools
else
  EXAMPLES_BIN=$CAFFE_BIN
  TOOLS_BIN=$CAFFE_BIN
fi  

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/cifar10_test_$DBTYPE

$EXAMPLES_BIN/convert_cifar_data.bin $DATA $EXAMPLE $DBTYPE

echo "Computing image mean..."

$TOOLS_BIN/compute_image_mean -backend=$DBTYPE \
  $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/mean.binaryproto

echo "Done."
