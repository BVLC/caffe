#!/bin/bash
# Usage:
# ./experiments/scripts/fast_rcnn.sh NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/fast_rcnn.sh VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

NET=$1
NET_lc=${NET,,}
DATASET=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=40000
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=280000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/fast_rcnn_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

if [[ ! -e "experiments/logs" ]]; then
    mkdir experiments/logs
fi

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py \
  --solver ../../models/intel_optimized_models/faster-rcnn/${PT_DIR}/${NET}/fast_rcnn/solver.prototxt \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py \
  --def ../../models/intel_optimized_models/faster-rcnn/${PT_DIR}/${NET}/fast_rcnn/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  ${EXTRA_ARGS}
