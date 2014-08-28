#!/usr/bin/env sh
# This scripts downloads the Caffe R-CNN ImageNet
# for ILSVRC13 detection.

MODEL=caffe_rcnn_imagenet_model
CHECKSUM=42c1556d2d47a9128c4a90e0a9c5341c

if [ -f $MODEL ]; then
  echo "Model already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $MODEL | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $MODEL | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Model checksum is correct. No need to download."
    exit 0
  else
    echo "Model checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading..."

wget http://dl.caffe.berkeleyvision.org/$MODEL examples/imagenet/$MODEL

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
