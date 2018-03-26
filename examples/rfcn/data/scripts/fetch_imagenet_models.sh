#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

FILE=imagenet_models.tgz
URL=https://dl.dropbox.com/s/gstw7122padlf0l/imagenet_models.tgz?dl=0
CHECKSUM=ed34ca912d6782edfb673a8c3a0bda6d

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading pretrained ImageNet models (1G)..."

wget $URL -O $FILE

echo "Unzipping..."

tar zxvf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
