#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=./build/examples/siamese
DATA=./data/mnist

$EXAMPLES/convert_mnist_siamese_images.bin \
    $DATA/train-images-idx3-ubyte \
    $DATA/train-labels-idx1-ubyte \
    ./examples/siamese/train_images

$EXAMPLES/convert_mnist_siamese_images.bin \
    $DATA/t10k-images-idx3-ubyte \
    $DATA/t10k-labels-idx1-ubyte \
    ./examples/siamese/test_images

echo "Done."
