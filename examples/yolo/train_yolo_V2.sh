#!/usr/bin/env sh

CAFFE_HOME=../..

SOLVER=$CAFFE_HOME/models/intel_optimized_models/yolo/yolov2_solver.prototxt

#SNAPSHOT
WEIGHTS=$CAFFE_HOME/models/intel_optimized_models/yolo/darknet19_448.caffemodel

$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --weights=$WEIGHTS 2>&1 | tee train_YOLO_V2.log
