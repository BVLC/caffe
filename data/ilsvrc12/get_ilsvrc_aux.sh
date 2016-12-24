#!/usr/bin/env sh
#
# N.B. This does not download the ilsvrcC12 data set, as it is gargantuan.
# This script downloads the imagenet example auxiliary files including:
# - the ilsvrc12 image mean, binaryproto
# - synset ids and words
# - Python pickle-format data of ImageNet graph structure and relative infogain
# - the training splits with labels

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

if ! [ -f ./det_synset_words.txt -a -f ./imagenet_mean.binaryproto -a -f ./synsets.txt \
        -a -f ./train.txt -a -f ./imagenet.bet.pickle -a -f ./synset_words.txt -a -f ./test.txt -a -f ./val.txt ]; then
        echo "Downloading..."
        wget -c http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
        echo "Unzipping..."
        tar -xf caffe_ilsvrc12.tar.gz && rm -f caffe_ilsvrc12.tar.gz
        echo "Done."
else
        echo "ImageNet example aux files already exist."
fi
