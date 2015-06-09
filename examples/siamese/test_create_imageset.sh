#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

BIN=./build/examples/siamese
SIAMESE=./examples/siamese

echo "Running full test of imageset creation"

echo "Create imageset from mnist database (requires mnist database!)"
$SIAMESE/create_siamese_images_from_mnist.sh

echo "Running full test of imageset creation"
echo "1a)  random no-limit: "
$BIN/create_imageset.bin \
    -i $SIAMESE/train_images \
    -o $SIAMESE/train_images/random_imageset.txt \
    -r
echo "1b)  random limit 10.000: "
$BIN/create_imageset.bin \
    -i $SIAMESE/test_images \
    -o $SIAMESE/test_images/random_10.000_imageset.txt \
    -r 10000
echo "2a)  full cross: "
$BIN/create_imageset.bin \
    -i $SIAMESE/test_images \
    -o $SIAMESE/test_images/full_cross_imageset.txt \
    -f
echo "2b)  limited cross (5, 50): "
$BIN/create_imageset.bin \
    -i $SIAMESE/train_images \
    -o $SIAMESE/train_images/cross_5_50_imageset.txt \
    -f 5 50
echo "3)  closest cross (5, 50): "
$BIN/create_imageset.bin \
    -i $SIAMESE/train_images \
    -o $SIAMESE/train_images/closest_5_50_imageset.txt \
    -c 5 50

echo "Creating leveldb..."
rm -rf $SIAMESE/siamese_imageset_closest_5_50_train_leveldb
rm -rf $SIAMESE/siamese_imageset_full_cross_test_leveldb

$EXAMPLES/convert_imageset_siamese_data.bin \
    $SIAMESE/train_images/closest_5_50_imageset.txt \
    $SIAMESE/siamese_imageset_closest_5_50_train_leveldb \
    1

$EXAMPLES/convert_imageset_siamese_data.bin \
    $SIAMESE/test_images/full_cross_imageset.txt \
    $SIAMESE/siamese_imageset_full_cross_test_leveldb \
    1

echo "Done."
