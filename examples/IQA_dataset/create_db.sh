#!/usr/bin/env sh
# Create the IQA dataset lmdb inputs
# N.B. set the path to the IQA dataset train + val data dirs
# First input argument is the number of database being generated
# Second input argument is the distortion type
set -e

DISTORTION_TYPE=$2
EXAMPLE=examples/IQA_dataset/$DISTORTION_TYPE
DATASET=live
DATA=data/$DATASET/$DISTORTION_TYPE
TOOLS=build/tools

TRAIN_DATA_ROOT=$DATA/
VAL_DATA_ROOT=$DATA/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_db.sh to the path" \
       "where the IQA training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_db.sh to the path" \
       "where the IQA validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/mappings/scores_train_$1.txt \
    $EXAMPLE/${DATASET}_${DISTORTION_TYPE}_train_$1_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $VAL_DATA_ROOT \
    $DATA/mappings/scores_val_$1.txt \
    $EXAMPLE/${DATASET}_${DISTORTION_TYPE}_val_$1_lmdb

echo "Done."
