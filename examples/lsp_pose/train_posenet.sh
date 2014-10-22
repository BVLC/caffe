#!/usr/bin/env sh

prefix=train-21-oct
postfix=finetune-from-10-oct

mkdir cache
mkdir cache/$prefix

TOOLS=../../build/tools

$TOOLS/caffe train -solver caffenet-pose-solver.prototxt -weights /home/wyang/github/caffe/examples/lsp_pose/cache/train-14-oct/train-14-oct-finetune-from-10-oct/pose_caffenet_train_iter_10000.caffemodel 2>&1 | tee cache/$prefix/$prefix-$postfix.log
