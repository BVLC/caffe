#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=./build/examples/triplet
DATA=./data/mnist

echo "Creating leveldb..."

rm -rf ./examples/triplet/mnist_siamese_train_leveldb
rm -rf ./examples/triplet/mnist_siamese_test_leveldb

$EXAMPLES/convert_mnist_triplet_data.bin \
    $DATA/train-images-idx3-ubyte \
    $DATA/train-labels-idx1-ubyte \
    ./examples/triplet/mnist_triplet_train_leveldb
$EXAMPLES/convert_mnist_triplet_data.bin \
    $DATA/t10k-images-idx3-ubyte \
    $DATA/t10k-labels-idx1-ubyte \
    ./examples/triplet/mnist_triplet_test_leveldb

echo "Done."
