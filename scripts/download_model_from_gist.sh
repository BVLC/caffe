#!/usr/bin/env sh

GIST=$1
DIRNAME=${2:-./models}

if [ -z $GIST ]; then
  echo "usage: download_model_from_gist.sh <gist_id> <dirname>"
  exit
fi

GIST_DIR=$(echo $GIST | tr '/' '-')
MODEL_DIR="$DIRNAME/$GIST_DIR"

if [ -d $MODEL_DIR ]; then
    echo "$MODEL_DIR already exists! Please make sure you're not overwriting anything important!"
    exit
fi

echo "Downloading Caffe model info to $MODEL_DIR ..."
mkdir -p $MODEL_DIR
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
wget https://gist.github.com/$GIST/download -O $MODEL_DIR/gist.zip
unzip -j $MODEL_DIR/gist.zip -d $MODEL_DIR
rm $MODEL_DIR/gist.zip
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
wget https://gist.github.com/$GIST/download -O $MODEL_DIR/gist.zip
unzip -j $MODEL_DIR/gist.zip -d $MODEL_DIR
rm $MODEL_DIR/gist.zip
=======
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
wget https://gist.github.com/$GIST/download -O $MODEL_DIR/gist.tar.gz
tar xzf $MODEL_DIR/gist.tar.gz --directory=$MODEL_DIR --strip-components=1
rm $MODEL_DIR/gist.tar.gz
>>>>>>> origin/BVLC/parallel
=======
wget https://gist.github.com/$GIST/download -O $MODEL_DIR/gist.zip
unzip -j $MODEL_DIR/gist.zip -d $MODEL_DIR
rm $MODEL_DIR/gist.zip
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
wget https://gist.github.com/$GIST/download -O $MODEL_DIR/gist.zip
unzip -j $MODEL_DIR/gist.zip -d $MODEL_DIR
rm $MODEL_DIR/gist.zip
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
echo "Done"
