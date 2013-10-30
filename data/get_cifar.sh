#!/usr/bin/env sh
# This scripts downloads the mnist data and unzips it.

echo "Downloading..."

wget -q http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

echo "Unzipping..."

tar xzf cifar-10-binary.tar.gz

echo "Done."
