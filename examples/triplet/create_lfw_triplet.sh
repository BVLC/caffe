#!/usr/bin/env sh
# This script converts the lfw data into leveldb format.

EXAMPLES=./build/examples/triplet
DATA=./data/lfw

echo "Creating leveldb..."

rm -rf ./examples/triplet/lfw_triplet_train_leveldb
rm -rf ./examples/triplet/lfw_triplet_test_leveldb

$EXAMPLES/convert_lfw_triplet_data.bin \
    $DATA/traindata \
    $DATA/trainlabel \
    ./examples/triplet/lfw_triplet_train_leveldb
$EXAMPLES/convert_lfw_triplet_data.bin \
    $DATA/testdata \
    $DATA/testlabel \
    ./examples/triplet/lfw_triplet_test_leveldb

echo "Done."
