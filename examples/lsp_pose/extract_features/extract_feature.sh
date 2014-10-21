#!/usr/bin/env sh
# args for EXTRACT_FEATURE
TOOL=../../../build/tools
MODEL=/home/wyang/github/caffe/examples/lsp_pose/pose_caffenet_train_iter_30000.caffemodel 
PROTOTXT=/home/wyang/github/caffe/examples/lsp_pose/caffenet-pose-lsp-train-val.prototxt

#--------------------------------------
# Step 1: Extract feature (LEVELDB)
# Layer=layer name
LAYER=conv1
# Result leveldb name
LEVELDB=features_${LAYER} 
# How many batch you want to extract
BATCHNUM=1
# Delete the previously extracted feature (LEVELMAP)
rm -r -f $LEVELDB
# Extract feature
$TOOL/extract_features.bin  $MODEL $PROTOTXT $LAYER $LEVELDB $BATCHNUM GPU

#--------------------------------------
# Step 2: LEVELDB -> Mat
# args for LEVELDB to MAT
BATCHSIZE=10
DIM=290400 # conv1
OUT=$LEVELDB/features.mat

sudo python ../../../tools/wyang/leveldb2mat.py $LEVELDB $BATCHNUM  $BATCHSIZE $DIM $OUT 
