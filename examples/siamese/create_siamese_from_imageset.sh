#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=./build/examples/siamese
echo "Creating leveldb..."

rm -rf ./examples/siamese/imageset_siamese_train_leveldb
rm -rf ./examples/siamese/imageset_siamese_test_leveldb

$EXAMPLES/convert_imageset_siamese_data.bin \
    ./examples/siamese/train_images/siamese_set.txt \
    ./examples/siamese/imageset_siamese_train_leveldb \
    1

$EXAMPLES/convert_imageset_siamese_data.bin \
    ./examples/siamese/test_images/siamese_set.txt \
    ./examples/siamese/imageset_siamese_test_leveldb \
    1

echo "Done."
