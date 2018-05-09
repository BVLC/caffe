#!/usr/bin/env sh

CAFFE_ROOT=../../..
ROOT_DIR=/your/path/to/vocroot/
LABEL_FILE=$CAFFE_ROOT/examples/yolo/data/label_map.txt

# 2007 + 2012 trainval
LIST_FILE=$CAFFE_ROOT/examples/yolo/data/trainval.txt
LMDB_DIR=./lmdb/trainval_lmdb
SHUFFLE=true

# 2007 test
# LIST_FILE=$CAFFE_ROOT/examples/yolo/data/test_2007.txt
# LMDB_DIR=./lmdb/test2007_lmdb
# SHUFFLE=false

RESIZE_W=448
RESIZE_H=448

#If you want to generate the test lmdb, please comment the trainval part and uncomment the test part
$CAFFE_ROOT/build/tools/convert_box_data --resize_width=$RESIZE_W --resize_height=$RESIZE_H \
  --label_file=$LABEL_FILE $ROOT_DIR $LIST_FILE $LMDB_DIR --encoded=true --encode_type=jpg --shuffle=$SHUFFLE

