#!/bin/bash

TOOLS=../../caffe/build/tools
WEIGHTS=TVG_CRFRNN_COCO_VOC.caffemodel
SOLVER=TVG_CRFRNN_new_solver.prototxt

$TOOLS/caffe train -solver $SOLVER -weights $WEIGHTS -gpu 0

