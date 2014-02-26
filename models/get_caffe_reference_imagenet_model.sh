#!/usr/bin/env sh
# This scripts downloads the caffe reference imagenet model
# for ilsvrc image classification and deep feature extraction

echo "Downloading..."

wget -q https://www.dropbox.com/s/n3jups0gr7uj0dv/caffe_reference_imagenet_model

echo "Done. Please check that the checksum = bf44bac4a59aa7792b296962fe483f2b."
