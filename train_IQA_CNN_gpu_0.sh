#!/bin/bash
# Trains IQA CNN on a range of databases provided as the first and second
# argument of this script using GPU 0
for i in $(seq $1 1 $2)
do
   ./build/tools/caffe train --solver=models/IQA_CNN/$i/solver.prototxt --gpu=0
done
