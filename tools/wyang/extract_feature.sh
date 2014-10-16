#!/usr/bin/env sh
# args for EXTRACT_FEATURE
TOOL=../../build/tools
MODEL=../../examples/imagenet/caffe_reference_imagenet_model
PROTOTXT=../../examples/_temp/imagenet_val.prototxt
LAYER=conv1
LEVELDB=../../examples/_temp/features_conv1
BATCHSIZE=10

# args for LEVELDB to MAT
DIM=290400
OUT=../../examples/_temp/features.mat
BATCHNUM=1

$TOOL/extract_features.bin  $MODEL $PROTOTXT $LAYER $LEVELDB $BATCHSIZE GPU
python leveldb2mat.py $LEVELDB $BATCHNUM  $BATCHSIZE $DIM $OUT 
