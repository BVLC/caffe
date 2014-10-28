#!/usr/bin/env sh
# This scripts downloads the CIFAR10 (binary version) data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

wget --no-check-certificate http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

echo "Unzipping..."

tar -xf cifar-10-binary.tar.gz && rm -f cifar-10-binary.tar.gz
mv cifar-10-batches-bin/* . && rm -rf cifar-10-batches-bin

# Creation is split out because leveldb sometimes causes segfault
# and needs to be re-created.

echo "Done."
