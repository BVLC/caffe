#!/usr/bin/env sh
# This scripts downloads the caffe reference imagenet model
# for ilsvrc image classification and deep feature extraction

MODEL=caffe_alexnet_model
CHECKSUM=91df0e19290ef78324de9eecb258a77f

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

wget --no-check-certificate https://www.dropbox.com/s/rk6nkt0kf109slo/caffe_alexnet_model

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
