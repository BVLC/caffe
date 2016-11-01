#!/bin/bash
# Create model and solver descriptions for multiple train-test iterations
# The number of train-test iterations to be done is provided as the first 
# argument to this script. The type of distortion is the second argument
N_ITERS=$1
DISTORTION_TYPE=$2
MODEL_DIR=models/IQA_CNN/$DISTORTION_TYPE
BATCH_SIZE=64
N_EPOCHS=40
if [ ! -d $MODEL_DIR ]; then
  mkdir $MODEL_DIR
fi
for i in $(seq 1 1 $N_ITERS)
do
  if [ ! -d $MODEL_DIR/$i ]; then
    mkdir $MODEL_DIR/$i
  fi
  # Read total number of training image patches
  N_TRAINING_INSTANCES=$(wc -l < "data/live/$DISTORTION_TYPE/mappings/scores_train_${i}.txt")
  # Calculate solver parameters which are based on the total number of training instances
  STEP_SIZE=$((N_TRAINING_INSTANCES/BATCH_SIZE))
  MAX_ITER=$((STEP_SIZE*N_EPOCHS))
  # Generate the solver.prototxt file  
  sed "1s/.*/net: \"models\/IQA_CNN\/$DISTORTION_TYPE\/$i\/train_val.prototxt\"/" models/IQA_CNN/solver_skeleton.prototxt > $MODEL_DIR/$i/solver.prototxt
  sed -i "7s/.*/stepsize: ${STEP_SIZE}/" $MODEL_DIR/$i/solver.prototxt
  sed -i "9s/.*/max_iter: ${MAX_ITER}/" $MODEL_DIR/$i/solver.prototxt
  sed -i "15s/.*/snapshot_prefix: \"models\/IQA_CNN\/$DISTORTION_TYPE\/$i\/IQA_CNN_train\"/" $MODEL_DIR/$i/solver.prototxt
  # Generate network description file
  sed "11s/.*/    source: \"examples\/IQA_dataset\/$DISTORTION_TYPE\/live_${DISTORTION_TYPE}_train_${i}_lmdb\"/" models/IQA_CNN/train_val.prototxt > $MODEL_DIR/$i/train_val.prototxt
  sed -i "25s/.*/    source: \"examples\/IQA_dataset\/$DISTORTION_TYPE\/live_${DISTORTION_TYPE}_val_${i}_lmdb\"/" $MODEL_DIR/$i/train_val.prototxt
done
