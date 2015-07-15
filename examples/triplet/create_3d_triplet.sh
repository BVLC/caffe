#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=./build/examples/triplet
DATA=./data/linemod

echo "Creating leveldb..."

rm -rf ./examples/triplet/mnist_3d_train_leveldb
rm -rf ./examples/triplet/mnist_3d_test_leveldb

$EXAMPLES/convert_3d_triplet_data.bin \
    $DATA/train-images-idx3-ubyte \
    $DATA/train-labels-idx1-ubyte \
    ./examples/triplet/3d_triplet_train_leveldb
$EXAMPLES/convert_3d_triplet_data.bin \
    $DATA/test-images-idx3-ubyte \
    $DATA/test-labels-idx1-ubyte \
    ./examples/triplet/3d_triplet_test_leveldb

echo "Done."
