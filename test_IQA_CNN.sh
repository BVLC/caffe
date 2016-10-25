#!/bin/bash
# Test/Validate the trained IQA CNN networks
# First argument is the number of networks to test/validate
# Second argument is the distortion type
OUTDIR=outputs
NITERS=$1
DISTORTION_TYPE=$2
if [ ! -d $OUTDIR ]; then
  mkdir $OUTDIR
fi
for i in $(seq 1 1 $NITERS)
do
   rm output_.txt
   rm scores_.txt
   ./build/tools/caffe test --weights=models/IQA_CNN/$DISTORTION_TYPE/$i/IQA_CNN_train_iter_33333.caffemodel --model=models/IQA_CNN/$DISTORTION_TYPE/$i/train_val.prototxt --iterations=12000 --gpu=all
   cp output_.txt $OUTDIR/output_$i.txt
   cp scores_.txt $OUTDIR/scores_$i.txt
done
   