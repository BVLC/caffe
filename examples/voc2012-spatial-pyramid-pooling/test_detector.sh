#!/usr/bin/env sh

TOOLS=../../build/tools
MODEL=spp_net_region_classifier_iter_

#    LOG(ERROR) << "test_net net_proto pretrained_net_proto iterations "
#        << "[CPU/GPU] [Device ID]";
GLOG_logtostderr=1 $TOOLS/test_net.bin spp_net_region_classifier_test.prototxt $MODEL \
  iterations

echo "Done."
