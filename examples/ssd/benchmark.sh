#!/usr/bin/env bash
SSD_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
$CAFFE_ROOT/build/tools/caffe time -model $SSD_ROOT/VGGNet/VOC0712/SSD_300x300/deploy.prototxt -gpu 0 -phase TEST -iterations 10 -lt
