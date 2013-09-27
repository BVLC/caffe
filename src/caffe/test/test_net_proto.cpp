// Copyright 2013 Yangqing Jia

#include <cstring>
#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>
#include <gtest/gtest.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
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
  FillerParameter filler_param;
  shared_ptr<Filler<TypeParam> > filler;
  filler.reset(new ConstantFiller<TypeParam>(filler_param));
  filler->Fill(label.get());
  filler.reset(new UniformFiller<TypeParam>(filler_param));
  filler->Fill(data.get());

  vector<Blob<TypeParam>*> bottom_vec;
  bottom_vec.push_back(data.get());
  bottom_vec.push_back(label.get());

  Net<TypeParam> caffe_net(net_param, bottom_vec);
  EXPECT_EQ(caffe_net.layer_names().size(), 9);
  EXPECT_EQ(caffe_net.blob_names().size(), 10);

  // Print a few statistics to see if things are correct
  for (int i = 0; i < caffe_net.blobs().size(); ++i) {
    LOG(ERROR) << "Blob: " << caffe_net.blob_names()[i];
    LOG(ERROR) << "size: " << caffe_net.blobs()[i]->num() << ", "
        << caffe_net.blobs()[i]->channels() << ", "
        << caffe_net.blobs()[i]->height() << ", "
        << caffe_net.blobs()[i]->width();
  }
  // Run the network without training.
  vector<Blob<TypeParam>*> top_vec;
  LOG(ERROR) << "Performing Forward";
  caffe_net.Forward(bottom_vec, &top_vec);
  LOG(ERROR) << "Performing Backward";
  LOG(ERROR) << caffe_net.Backward();

}

}  // namespace caffe
