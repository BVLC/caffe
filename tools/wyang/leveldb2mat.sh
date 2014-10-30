#!/usr/bin/env sh

LEVELDB=../../examples/_temp/features_conv1
DIM=290400
OUT=../../examples/_temp/features.mat
BATCH_NUM=1
BATCH_SIZE=10

python leveldb2mat.py $LEVELDB $BATCH_NUM  $BATCH_SIZE $DIM $OUT 
