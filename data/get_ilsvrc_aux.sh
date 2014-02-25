#!/usr/bin/env sh
# This scripts downloads the ilsvrc auxiliary files including
# - the ilsvrc image mean, binaryproto
# - synset ids and words
# - the training splits with labels

echo "Downloading..."

wget -q https://www.dropbox.com/s/1cyhk5k5kjcfq92/caffe_ilsvrc_2012.tar.gz

echo "Unzipping..."

tar -xf caffe_ilsvrc_2012.tar.gz && rm -f caffe_ilsvrc_2012.tar.gz

echo "Done."
