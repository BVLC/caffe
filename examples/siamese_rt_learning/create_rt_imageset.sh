#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

BIN=./build/examples/siamese_rt_learning
EXAMPLE=./examples/siamese_rt_learning
SIAMESE=./examples/siamese


$BIN/create_rt_imageset.bin \
    -i $SIAMESE/train_images \
    -o $EXAMPLE/test_file.txt \
    -c 5 5 \
    -e .pgm
