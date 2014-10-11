#!/usr/bin/env sh
# args for EXTRACT_FEATURE
TOOL=../../../build/tools
MODEL=/home/wyang/Code/caffe-master/examples/lsp_pose_h5/pose_caffenet_train_iter_12000.caffemodel
PROTOTXT=../caffenet-pose-lsp-train-val.prototxt
# CONV 1
#LAYER=conv1
#LEVELDB=features_${LAYER}_0923

#rm -r -f $LEVELDB
# FC8-POSE
LAYER=fc8
LEVELDB=pred-09-oct-1024

# LABEL
#LAYER=label
#LEVELDB=groundtruth_0310

BATCHSIZE=100

# args for LEVELDB to MAT
#DIM=290400 # conv1
#DIM=186624 # conv2
#DIM=64896 # conv3
DIM=28 # fc8-pose
OUT=$LEVELDB/features.mat
BATCHNUM=10

rm -rf $LEVELDB

$TOOL/extract_features.bin  $MODEL $PROTOTXT $LAYER $LEVELDB $BATCHNUM GPU
sudo python ../../../build/wyang/leveldb2mat.py $LEVELDB $BATCHNUM  $BATCHSIZE $DIM $OUT 
