#!/usr/bin/env sh

prefix=train-16-oct
postfix=finetune-remove-pool5

mkdir cache
mkdir cache/$prefix

TOOLS=../../build/tools

$TOOLS/caffe train -solver caffenet-pose-solver.prototxt -weights /home/wyang/github/caffe/examples/lsp_pose/cache/train-15-oct/train-15-oct-finetune-from-10-oct-remove-pool5/pose_caffenet_train_iter_58000.caffemodel 2>&1 | tee cache/$prefix/$prefix-$postfix.log
