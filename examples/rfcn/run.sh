#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export GLOG_minloglevel=0


unset OMP_NUM_THREADS
# export OMP_NUM_THREADS=44
# export MKL_NUM_THREADS

# export OMP_WAIT_POLICY=passive
unset MKL_THREADING_LAYER
# export MKL_THREADING_LAYER=gnu


./experiments/scripts/rfcn_end2end.sh 0 ResNet-101 pascal_voc
