#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_alt_opt.sh NET DATASET [options args to {train,test}_net.py]
# DATASET is only pascal_voc for now
#
# Example:
# ./experiments/scripts/faster_rcnn_alt_opt.sh VGG_CNN_M_1024 pascal_voc \
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
    echo "Not implemented: use experiments/scripts/faster_rcnn_end2end.sh for coco"
    exit
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/faster_rcnn_alt_opt_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

if [[ ! -e "experiments/logs" ]]; then
    mkdir experiments/logs
fi

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_faster_rcnn_alt_opt.py \
  --net_name ${NET} \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_alt_opt.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep "Final model:" ${LOG} | awk '{print $3}'`
set -x

time ./tools/test_net.py \
  --def ../../models/intel_optimized_models/faster-rcnn/${PT_DIR}/${NET}/faster_rcnn_alt_opt/faster_rcnn_test.pt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_alt_opt.yml \
  ${EXTRA_ARGS}
