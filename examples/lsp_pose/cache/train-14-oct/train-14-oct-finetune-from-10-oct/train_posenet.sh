#!/usr/bin/env sh

prefix=train-14-oct
postfix=finetune-from-10-oct

mkdir cache
mkdir cache/$prefix

TOOLS=../../build/tools

$TOOLS/caffe train -solver caffenet-pose-solver.prototxt -weights /home/wyang/github/caffe/examples/lsp_pose/cache/train-10-oct/train-10-oct-conv-fix-val-mean-fc-shuffle/pose_caffenet_train_iter_70000.caffemodel 2>&1 | tee cache/$prefix/$prefix-$postfix.log
