#!/usr/bin/env sh
# This script converts the cifar data into leveldb or lmdb format.

EXAMPLE=examples/cifar10
DATA=data/cifar10
BUILD=./build/examples/cifar10
TOOLS=./build/tools

# DBLIST contains the available DB types for the CIFAR-10 dataset.
# Update the list if more DB types are supported.
DBLIST=(lmdb leveldb)
# DBTYPE that is choosen. Can be leveldb or lmdb.
DBTYPE=leveldb

echo "Creating $DBTYPE..."

for type in "${DBLIST[@]}"
do
  rm -rf $EXAMPLE/cifar10_train_$type
  rm -rf $EXAMPLE/cifar10_test_$type
done

$BUILD/convert_cifar_data.bin $DATA $EXAMPLE $DBTYPE

echo "Computing image mean..."

$TOOLS/compute_image_mean.bin -backend=$DBTYPE $EXAMPLE/cifar10_train_$DBTYPE \
  $EXAMPLE/mean.binaryproto

echo "Done."
