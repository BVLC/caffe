// Copyright 2013 Yangqing Jia

#include <cuda_runtime.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <gtest/gtest.h>

#include <cstring>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class NetProtoTest : public ::testing::Test {};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(NetProtoTest, Dtypes);

TYPED_TEST(NetProtoTest, TestSetup) {
  NetParameter net_param;
  ReadProtoFromTextFile("caffe/test/data/lenet.prototxt", &net_param);
  // check if things are right
  EXPECT_EQ(net_param.layers_size(), 10);
  EXPECT_EQ(net_param.input_size(), 0);

  vector<Blob<TypeParam>*> bottom_vec;

  Net<TypeParam> caffe_net(net_param, bottom_vec);
  EXPECT_EQ(caffe_net.layer_names().size(), 10);
  EXPECT_EQ(caffe_net.blob_names().size(), 10);

  // Print a few statistics to see if things are correct
  for (int i = 0; i < caffe_net.blobs().size(); ++i) {
    LOG(ERROR) << "Blob: " << caffe_net.blob_names()[i];
    LOG(ERROR) << "size: " << caffe_net.blobs()[i]->num() << ", "
        << caffe_net.blobs()[i]->channels() << ", "
        << caffe_net.blobs()[i]->height() << ", "
        << caffe_net.blobs()[i]->width();
  }
  Caffe::set_mode(Caffe::CPU);
  // Run the network without training.
  LOG(ERROR) << "Performing Forward";
  caffe_net.Forward(bottom_vec);
  LOG(ERROR) << "Performing Backward";
  LOG(ERROR) << caffe_net.Backward();
  
  Caffe::set_mode(Caffe::GPU);
  // Run the network without training.
  LOG(ERROR) << "Performing Forward";
  caffe_net.Forward(bottom_vec);
  LOG(ERROR) << "Performing Backward";
  LOG(ERROR) << caffe_net.Backward();
}

}  // namespace caffe
