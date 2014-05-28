#!/usr/bin/env sh
# This scripts downloads the caffe reference imagenet model
# for ilsvrc image classification and deep feature extraction

MODEL=caffe_reference_imagenet_model
CHECKSUM=bf44bac4a59aa7792b296962fe483f2b

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

wget --no-check-certificate https://www.dropbox.com/s/n3jups0gr7uj0dv/$MODEL

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
