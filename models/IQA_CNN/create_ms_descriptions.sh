#!/bin/bash
# Create model and solver descriptions for multiple train-test iterations
# The number of train-test iterations to be done is provided as the first 
# argument to this script. The type of distortion is the second argument
N_ITERS=$1
DISTORTION_TYPE=$2
MODEL_DIR=models/IQA_CNN/$DISTORTION_TYPE
for i in $(seq 1 1 $N_ITERS)
do
#  mkdir $MODEL_DIR/$i
  sed '1s/.*/net: "modelsIQA_CNN${i}train_val.prototxt"/' models/IQA_CNN/solver.prototxt > $MODEL_DIR/$i/solver.prototxt
#  sed '1s/.*/net: "models/IQA_CNN/$DISTORTION_TYPE/$i/train_val.prototxt"/' models/IQA_CNN/solver.prototxt > $MODEL_DIR/$i/solver.prototxt
#  sed -i '15s/.*/snapshot_prefix: "models/IQA_CNN/$DISTORTION_TYPE/$i/IQA_CNN_train"/' $MODEL_DIR/$i/solver.prototxt
#  sed '11s/.*/    source: "examples/IQA_dataset/$DISTORTION_TYPE/live_$DISTORTION_TYPE_train_$i_lmdb"/' models/IQA_CNN/train_val.prototxt > $MODEL_DIR/$i/train_val.prototxt
#  sed -i '25s/.*/    source: "examples/IQA_dataset/$DISTORTION_TYPE/live_$DISTORTION_TYPE_val_$i_lmdb"/' $MODEL_DIR/$i/train_val.prototxt
done
