#!/usr/bin/env bash

GPU_ID=0
WEIGHTS=\
./examples/coco_caption/lrcn_iter_110000.caffemodel
DATA_DIR=./examples/coco_caption/h5_data/
if [ ! -d $DATA_DIR ]; then
    echo "Data directory not found: $DATA_DIR"
    echo "First, download the COCO dataset (follow instructions in data/coco)"
    echo "Then, run ./examples/coco_caption/coco_to_hdf5_data.py to create the Caffe input data"
    exit 1
fi

./build/tools/caffe train \
    -solver ./examples/coco_caption/lrcn_finetune_solver.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
