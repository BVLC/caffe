#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

echo "Creating leveldb..."

rm -rf mnist-train-leveldb
rm -rf mnist-test-leveldb

../build/examples/convert_mnist_data.bin train-images-idx3-ubyte train-labels-idx1-ubyte mnist-train-leveldb
../build/examples/convert_mnist_data.bin t10k-images-idx3-ubyte t10k-labels-idx1-ubyte mnist-test-leveldb

echo "Done."