#!/usr/bin/env sh

prefix=train-10-oct
postfix=conv-fix-val-mean-fc-shuffle

mkdir cache
mkdir cache/$prefix

TOOLS=../../build/tools

#$TOOLS/caffe train -solver caffenet-pose-solver.prototxt -weights /home/wyang/Code/caffe-master/examples/lsp_pose_h5/cache/train-09-oct/val-mean-4096-fix-conv-model/pose_caffenet_train_iter_13000.caffemodel 2>&1 | tee cache/$prefix/$prefix-$postfix.log

$TOOLS/caffe train -solver caffenet-pose-solver.prototxt -weights /home/wyang/Code/alex_finetune_ilsvrc14_gt_sel_6per_1000_pretrain_box/finetune_ilsvrc13_val1_sel_6per_iter_83000 2>&1 | tee cache/$prefix/$prefix-$postfix.log
