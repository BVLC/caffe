#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLE=examples/mnist
DATA=data/mnist
BUILD=build/examples/mnist

echo "Creating leveldb..."

rm -rf mnist_train_leveldb
rm -rf mnist_test_leveldb

$BUILD/convert_mnist_data.bin $DATA/train-images-idx3-ubyte \
  $DATA/train-labels-idx1-ubyte $EXAMPLE/mnist_train_leveldb
$BUILD/convert_mnist_data.bin $DATA/t10k-images-idx3-ubyte \
  $DATA/t10k-labels-idx1-ubyte $EXAMPLE/mnist_test_leveldb

echo "Done."
