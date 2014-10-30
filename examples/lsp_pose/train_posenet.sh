#!/usr/bin/env sh

prefix=train-29-oct
postfix=finetune

mkdir cache
mkdir cache/$prefix

TOOLS=../../build/tools

$TOOLS/caffe train -solver caffenet-pose-solver.prototxt -weights /home/wyang/github/caffe/examples/lsp_pose/cache/train-28-oct/train-28-oct-finetune/pose_caffenet_train_iter_30000.caffemodel 2>&1 | tee cache/$prefix/$prefix-$postfix.log
