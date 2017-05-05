#!/bin/sh
export CAFFE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/../..
export PYTHONPATH=$CAFFE_ROOT/"python"
# generate new fused model
python $CAFFE_ROOT/tools/inference-optimize/model_fuse.py \
  --indefinition $CAFFE_ROOT/models/bvlc_googlenet/deploy.prototxt \
  --inmodel $CAFFE_ROOT/models/bvlc_googlenet/bvlc_googlenet.caffemodel \
  --outdefinition $CAFFE_ROOT/models/bvlc_googlenet/fused_deploy.prototxt \
  --outmodel $CAFFE_ROOT/models/bvlc_googlenet/fused_bvlc_googlenet.caffemodel \
  --half_precision_mode=HALF_NONE \

#Use cpp_classfication to test
$CAFFE_ROOT/build/examples/cpp_classification/classification.bin \
  $CAFFE_ROOT/models/bvlc_googlenet/fused_deploy.prototxt \
  $CAFFE_ROOT/models/bvlc_googlenet/fused_bvlc_googlenet.caffemodel \
  $CAFFE_ROOT/data/ilsvrc12/imagenet_mean.binaryproto \
  $CAFFE_ROOT/data/ilsvrc12/synset_words.txt \
  $CAFFE_ROOT/examples/images/cat.jpg


