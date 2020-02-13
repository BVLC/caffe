#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

BIN=./build/examples/siamese
DATA=./examples/siamese
OUTPUT=./examples/siamese

echo "Creating leveldb..."

#$BIN/create_imageset.bin \
#    -i $DATA/train_images \
#    -o $OUTPUT/train_images/closest_cross_train_imageset.txt \
#    -c 5 50

$BIN/create_imageset.bin \
    -i $DATA/test_images \
    -o $OUTPUT/test_images/limit_cross_100_1000_test_imageset.txt \
    -f 100 1000

echo "Done."
