#!/usr/bin/env sh
# This scripts downloads the mnist data and unzips it.

echo "Downloading..."

wget -q http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -q http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -q http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -q http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

echo "Unzipping..."

gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz

echo "Creating leveldb..."

../build/examples/convert_mnist_data.bin train-images-idx3-ubyte train-labels-idx1-ubyte mnist-train-leveldb
../build/examples/convert_mnist_data.bin t10k-images-idx3-ubyte t10k-labels-idx1-ubyte mnist-test-leveldb

echo "Done."
