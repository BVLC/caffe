#!/usr/bin/env sh

TOOLS=../../build/tools

if [ -z "$1" ]; then
  echo "Using CPU! To run GPU speedtest, use:"
  echo "    ./speedtest_imagenet.sh <device ID>"
  echo "(Try ./speedtest_imagenet.sh 0 if you have just one GPU.)"
  sleep 3  # Let the user read
  WITH_GPU=false
  DEVICE_ID=0
else
  WITH_GPU=true
  DEVICE_ID=$1
fi

$TOOLS/caffe time \
  --model=imagenet_train_val.prototxt \
  --gpu=${WITH_GPU} \
  --device_id=${DEVICE_ID}

echo "Done."
