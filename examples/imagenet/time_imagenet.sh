#!/usr/bin/env sh

if [ -z "$1" ]; then
  echo "Using CPU! To time GPU mode, use:"
  echo "    ./time_imagenet.sh <device ID>"
  echo "(Try ./time_imagenet.sh 0 if you have just one GPU.)"
  sleep 3  # Let the user read
  GPU=""
else
  GPU="--gpu=$1"
fi

./build/tools/caffe time \
  --model=examples/imagenet/imagenet_train_val.prototxt \
  ${GPU}

echo "Done."
