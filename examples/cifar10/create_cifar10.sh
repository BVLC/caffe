#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLE=examples/cifar10
DATA=data/cifar10

echo "Creating leveldb..."

rm -rf $EXAMPLE/cifar10_train_leveldb $EXAMPLE/cifar10_test_leveldb

./build/examples/cifar10/convert_cifar_data.bin $DATA $EXAMPLE

echo "Computing image mean..."

./build/tools/compute_image_mean $EXAMPLE/cifar10_train_leveldb \
  $EXAMPLE/mean.binaryproto leveldb

echo "Done."
