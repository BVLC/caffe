#!/usr/bin/env bash
export CAFFE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/../..
export PYTHONPATH=$CAFFE_ROOT/"python"
# WHERE OU WISH TO STORE THE IMAGE DATASET AND LMDB AND SCORING RESULTS
export DATAPATH="/home/sfraczek/disk/sfraczek/ssd"
echo CAFFE_ROOT is $CAFFE_ROOT
echo PYTHONPATH is $PYTHONPATH
echo DATAPATH is $DATAPATH
