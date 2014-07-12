#!/usr/bin/env sh

TOOLS=../../build/tools
#MODEL=../imagenet/caffe_reference_imagenet_model
MODEL=voc2012_finetune_imagenet_spatial_pyramid_pooling_iter_2500
# MODEL=voc2012_finetune_spatial_pyramid_pooling_iter_6000
    
#    "This program takes in a trained network and an input data layer, and then"
#    " extract features of the input data produced by the net.\n"
#    "Usage: extract_features  pretrained_net_param"
#    "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
#    "  save_feature_leveldb_name1[,name2,...]  num_mini_batches  [CPU/GPU]"
#    "  [DEVICE_ID=0] [sample_keys_file]\n"

GLOG_logtostderr=1 $TOOLS/extract_features.bin $MODEL spp_net_conv5_train.prototxt \
  conv5 voc2012_spp_net_conv5_train.leveldb 58 CPU 0 voc2012_sample_keys_train.txt
  
#GLOG_logtostderr=1 $TOOLS/extract_features.bin $MODEL spp_net_conv5_test.prototxt \
#  conv5 voc2012_spp_net_conv5_test.leveldb 59 CPU 0 voc2012_sample_keys_test.txt

echo "Done."
