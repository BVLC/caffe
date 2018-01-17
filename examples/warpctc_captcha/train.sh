#!/bin/bash
GPU_ID=0
cur_date=`date +%Y-%m-%d-%H-%M-%S`
log_file_name=./examples/warpctc_captcha/log/${cur_date}
./build/tools/caffe train \
    -solver ./examples/warpctc_captcha/solver.prototxt \
    -gpu $GPU_ID 2>&1 | tee -a ${log_file_name}
