#!/bin/bash
# Test/Validate the trained IQA CNN networks
# First argument is the number of networks to test/validate
for i in $(seq 1 1 $1)
do
   rm output_.txt
   rm scores_.txt
   ./build/tools/caffe test --weights=models/IQA_CNN/$i/IQA_CNN_train_iter_33333.caffemodel --model=models/IQA_CNN/$i/train_val.prototxt --iterations=12000 --gpu=all
   cp output_.txt outputs/output_$i.txt
   cp scores_.txt outputs/scores_$i.txt
done
   