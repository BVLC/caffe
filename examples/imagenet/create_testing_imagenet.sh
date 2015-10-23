#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet test data dir

EXAMPLE=examples/imagenet
DATA=data/ilsvrc12
TOOLS=build/tools

# TRAIN_DATA_ROOT=/path/to/imagenet/train/
# VAL_DATA_ROOT=/path/to/imagenet/val/
TEST_DATA_ROOT=/home/wew57/imagenet-ILSVRC2012/testing/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

#if [ ! -d "$TRAIN_DATA_ROOT" ]; then
#  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
#  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
#       "where the ImageNet training data is stored."
#  exit 1
#fi
#
#if [ ! -d "$VAL_DATA_ROOT" ]; then
#  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
#  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
#       "where the ImageNet validation data is stored."
#  exit 1
#fi

if [ ! -d "$TEST_DATA_ROOT" ]; then
  echo "Error: TEST_DATA_ROOT is not a path to a directory: $TEST_DATA_ROOT"
  exit 1
fi

echo "Creating test lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TEST_DATA_ROOT \
    $DATA/test.txt \
    $EXAMPLE/ilsvrc12_test_lmdb

echo "Done."
