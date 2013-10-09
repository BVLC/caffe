// Copyright 2013 Yangqing Jia

#include <gtest/gtest.h>

#include <cstring>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class NetProtoTest : public ::testing::Test {};

typedef ::testing::Types<float> Dtypes;
TYPED_TEST_CASE(NetProtoTest, Dtypes);

TYPED_TEST(NetProtoTest, TestLoadFromText) {
  NetParameter net_param;
  ReadProtoFromTextFile("data/simple_conv.prototxt", &net_param);
  Blob<TypeParam> lena_image;
  ReadImageToBlob<TypeParam>(string("data/lena_256.jpg"), &lena_image);
  vector<Blob<TypeParam>*> bottom_vec;
  bottom_vec.push_back(&lena_image);

  for (int i = 0; i < lena_image.count(); ++i) {
    EXPECT_GE(lena_image.cpu_data()[i], 0);
    EXPECT_LE(lena_image.cpu_data()[i], 1);
  }

  Caffe::set_mode(Caffe::CPU);
  // Initialize the network, and then does smoothing
  Net<TypeParam> caffe_net(net_param, bottom_vec);
  LOG(ERROR) << "Start Forward.";
  const vector<Blob<TypeParam>*>& output = caffe_net.Forward(bottom_vec);
  LOG(ERROR) << "Forward Done.";
  EXPECT_EQ(output[0]->num(), 1);
  EXPECT_EQ(output[0]->channels(), 1);
  EXPECT_EQ(output[0]->height(), 252);
  EXPECT_EQ(output[0]->width(), 252);
  for (int i = 0; i < output[0]->count(); ++i) {
    EXPECT_GE(output[0]->cpu_data()[i], 0);
    EXPECT_LE(output[0]->cpu_data()[i], 1);
  }
  WriteBlobToImage<TypeParam>(string("lena_smoothed.png"), *output[0]);
}


}  // namespace caffe
