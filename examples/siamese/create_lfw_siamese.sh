#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=./build/examples/siamese
DATA=./data/lfw

echo "Creating leveldb..."

rm -rf ./examples/siamese/lfw_siamese_train_leveldb
rm -rf ./examples/siamese/lfw_siamese_test_leveldb

$EXAMPLES/convert_lfw_siamese_data.bin \
    $DATA/traindata \
    $DATA/trainlabel \
    ./examples/siamese/lfw_siamese_train_leveldb
$EXAMPLES/convert_mnist_siamese_data.bin \
    $DATA/testdata \
    $DATA/testlabel \
    ./examples/siamese/lfw_siamese_test_leveldb

echo "Done."
