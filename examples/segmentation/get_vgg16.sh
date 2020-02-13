#!/usr/bin/env sh
# This scripts downloads the VGG16 weights.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

FN_WG=VGG_ILSVRC_16_layers.caffemodel

if [ ! -e $FN_WG ]; then
  wget --no-check-certificate http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/${FN_WG}
fi
