// Copyright 2013 Yangqing Jia

#include <cstring>
#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>
#include <gtest/gtest.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/test/lenet.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class NetProtoTest : public ::testing::Test {};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(NetProtoTest, Dtypes);

TYPED_TEST(NetProtoTest, TestSetup) {
  NetParameter net_param;
  string lenet_string(kLENET);
  // Load the network
  CHECK(google::protobuf::TextFormat::ParseFromString(
      lenet_string, &net_param));
  // check if things are right
  EXPECT_EQ(net_param.layers_size(), 9);
  EXPECT_EQ(net_param.bottom_size(), 2);
  EXPECT_EQ(net_param.top_size(), 0);

  // Now, initialize a network using the parameter
  shared_ptr<Blob<TypeParam> > data(new Blob<TypeParam>(10, 1, 28, 28));
  shared_ptr<Blob<TypeParam> > label(new Blob<TypeParam>(10, 1, 1, 1));
  vector<Blob<TypeParam>*> bottom_vec;
  bottom_vec.push_back(data.get());
  bottom_vec.push_back(label.get());

  Net<TypeParam> caffe_net(net_param, bottom_vec);
  EXPECT_EQ(caffe_net.layer_names().size(), 9);
  EXPECT_EQ(caffe_net.blob_names().size(), 10);
}

}  // namespace caffe
