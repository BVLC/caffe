#!/bin/bash
# Usage:
# ./experiments/scripts/rfcn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/rfcn_end2end.sh 0 ResNet50 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

# GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_0712_trainval"
    TEST_IMDB="voc_0712_test"
    PT_DIR="pascal_voc"
    ITERS=110000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_val"
    PT_DIR="coco"
    ITERS=960000
    ;;
  luna)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="luna_2016_train"
    TEST_IMDB="luna_2016_val"
    PT_DIR="luna"
    ITERS=960000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/rfcn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

if [[ ! -e "experiments/logs" ]]; then
    mkdir experiments/logs
fi

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py  \
  --solver ../../models/intel_optimized_models/rfcn/${PT_DIR}/${NET}/rfcn_end2end/solver.prototxt \
  --imdb ${TRAIN_IMDB} \
  --weights data/imagenet_models/${NET}-model.caffemodel \
  --iters ${ITERS} \
  --cfg experiments/cfgs/rfcn_end2end.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py  \
  --def ../../models/intel_optimized_models/rfcn/${PT_DIR}/${NET}/rfcn_end2end/test_agnostic.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/rfcn_end2end.yml \
  ${EXTRA_ARGS}
