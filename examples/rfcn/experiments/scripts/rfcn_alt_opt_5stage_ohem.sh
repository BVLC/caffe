#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_alt_opt.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is only pascal_voc for now
#
# Example:
# ./experiments/scripts/faster_rcnn_alt_opt.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
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
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_val"
    PT_DIR="coco"
    ITERS=40000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/rfcn_alt_opt_5step_ohem_rpn4_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
if [[ ! -e "experiments/logs" ]]; then
    mkdir experiments/logs
fi
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_rfcn_alt_opt_5stage.py --gpu ${GPU_ID} \
  --net_name ${NET} \
  --weights data/imagenet_models/${NET}-model.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --imdb_test ${TEST_IMDB} \
  --cfg experiments/cfgs/rfcn_alt_opt_5step_ohem.yml \
  --model 'rfcn_alt_opt_5step_ohem'
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep "Final model" ${LOG} | awk '{print $3}'`
RPN_FINAL=`grep "Final RPN" ${LOG} | awk '{print $3}'`
set -x
time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/rfcn_alt_opt_5step_ohem/rfcn_test.pt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --rpn_file ${RPN_FINAL} \
  --cfg experiments/cfgs/rfcn_alt_opt_5step_ohem.yml \
  --num_dets 400
  ${EXTRA_ARGS}
