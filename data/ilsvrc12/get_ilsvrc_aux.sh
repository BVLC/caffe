#!/usr/bin/env sh
#
# N.B. This does not download the ilsvrcC12 data set, as it is gargantuan.
# This script downloads the imagenet example auxiliary files including:
# - the ilsvrc12 image mean, binaryproto
# - synset ids and words
# - the training splits with labels

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

wget --no-check-certificate https://www.dropbox.com/s/g5myor4y2scdv95/caffe_ilsvrc12.tar.gz

echo "Unzipping..."

tar -xf caffe_ilsvrc12.tar.gz && rm -f caffe_ilsvrc12.tar.gz

echo "Done."
