#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=../../build/examples/lenet
DATA=../../data/mnist

echo "Creating leveldb..."

rm -rf mnist-train-leveldb
rm -rf mnist-test-leveldb

$EXAMPLES/convert_mnist_data.bin $DATA/train-images-idx3-ubyte train-labels-idx1-ubyte mnist-train-leveldb
$EXAMPLES/convert_mnist_data.bin $DATA/t10k-images-idx3-ubyte t10k-labels-idx1-ubyte mnist-test-leveldb

echo "Done."
