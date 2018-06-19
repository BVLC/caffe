#!/usr/bin/env sh

CAFFE_HOME=../..

PROTO=$CAFFE_HOME/models/intel_optimized_models/yolo/yolov2_test.prototxt
MODEL=$CAFFE_HOME/models/intel_optimized_models/yolo/yolov2_iter_80000.caffemodel
ITER=619

echo $MODEL

$CAFFE_HOME/build/tools/test_detection \
    --model=$PROTO --iterations=$ITER \
    --weights=$MODEL --objects=5 --classes=20 --side=17