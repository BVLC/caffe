#!/bin/bash
# Trains IQA CNN on a range of databases provided as the first and second
# argument of this script using GPU 1
# Third argument is the distortion type
ITER_START=$1
ITER_END=$2
DISTORTION_TYPE=$3
for i in $(seq $ITER_START 1 $ITER_END)
do
   ./build/tools/caffe train --solver=models/IQA_CNN/$DISTORTION_TYPE/$i/solver.prototxt --gpu=1
   mv models/IQA_CNN/$DISTORTION_TYPE/$i/*.caffemodel models/IQA_CNN/$DISTORTION_TYPE/$i/IQA_CNN.caffemodel
   chmod a+r models/IQA_CNN/$DISTORTION_TYPE/$i/IQA_CNN.caffemodel
done
