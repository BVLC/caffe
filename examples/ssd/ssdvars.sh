#!/usr/bin/env bash
export CAFFE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/../..
export PYTHONPATH=$CAFFE_ROOT/"python"

# datapath is where you wish to store generated lmdb
# also in $DATAPATH/data directory you should have unpacked
# VOCdevkit and/or coco directories. see more in ./data/coco/README.md
# this variable is used in create_list and create_data scripts only.
export DATAPATH="/home/data/ssd/"

echo CAFFE_ROOT is $CAFFE_ROOT
echo PYTHONPATH is $PYTHONPATH
echo DATAPATH is $DATAPATH
