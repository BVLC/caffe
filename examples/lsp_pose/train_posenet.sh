#!/usr/bin/env sh

prefix=train-17-oct
postfix=finetune-conv-remove-pool5

mkdir cache
mkdir cache/$prefix

TOOLS=../../build/tools

$TOOLS/caffe train -solver caffenet-pose-solver.prototxt -weights /home/wyang/github/caffe/examples/lsp_pose/cache/train-16-oct/train-16-oct-finetune-remove-pool5/pose_caffenet_train_iter_36000.caffemodel 2>&1 | tee cache/$prefix/$prefix-$postfix.log
