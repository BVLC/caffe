#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=./build/examples/triplet
DATA=./data/linemod

echo "Creating leveldb..."

rm -rf ./examples/triplet/3d_triplet_train_leveldb
rm -rf ./examples/triplet/3d_triplet_test_leveldb

$EXAMPLES/convert_3d_triplet_data.bin \
    $DATA/binary_image_train \
    $DATA/binary_label_train \
    ./examples/triplet/3d_triplet_train_leveldb
$EXAMPLES/convert_3d_triplet_data.bin \
    $DATA/binary_image_test \
    $DATA/binary_label_test \
    ./examples/triplet/3d_triplet_test_leveldb

echo "Done."
