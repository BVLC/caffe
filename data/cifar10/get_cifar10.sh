#!/usr/bin/env sh
# This scripts downloads the CIFAR10 (binary version) data and unzips it.

DIR="$(readlink -f $(dirname "$0"))"
cd $DIR

echo "Downloading..."

wget -q http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

echo "Unzipping..."

tar -xf cifar-10-binary.tar.gz && rm -f cifar-10-binary.tar.gz
mv cifar-10-batches-bin/* . && rm -rf cifar-10-batches-bin

# Creation is split out because leveldb sometimes causes segfault
# and needs to be re-created.

echo "Done."
