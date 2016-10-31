#!/bin/bash
# Test/Validate the trained IQA CNN networks 
# First argument and Second arguments provide the range of the networks to test/validate
# Third argument is the distortion type
OUTDIR=outputs
ITER_START=$1
ITER_END=$2
DISTORTION_TYPE=$3
if [ ! -d $OUTDIR ]; then
  mkdir $OUTDIR
fi
for i in $(seq $ITER_START 1 $ITER_END)
do
   rm output_.txt
   rm scores_.txt
   ./build/tools/caffe test --weights=models/IQA_CNN/$DISTORTION_TYPE/$i/IQA_CNN.caffemodel --model=models/IQA_CNN/$DISTORTION_TYPE/$i/train_val.prototxt --iterations=12000 --gpu=all
   cp output_.txt $OUTDIR/output_$i.txt
   cp scores_.txt $OUTDIR/scores_$i.txt
done
chmod -R a+rw $OUTDIR
