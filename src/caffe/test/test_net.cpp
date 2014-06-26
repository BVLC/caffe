// Copyright 2014 BVLC and contributors.

#include <google/protobuf/text_format.h>
#include <string>

#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class NetTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    const string& proto =
        "name: 'TestNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    num: 5 "
        "    channels: 1 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "  top: 'label' "
        "} "
        "layers: { "
        "  name: 'innerproduct' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 1000 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "  } "
        "  blobs_lr: 1. "
        "  blobs_lr: 2. "
        "  weight_decay: 1. "
        "  weight_decay: 0. "
        "  bottom: 'data' "
        "  top: 'innerproduct' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: SOFTMAX_LOSS "
        "  bottom: 'innerproduct' "
        "  bottom: 'label' "
        "  top: 'top_loss' "
        "} ";
    NetParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    net_.reset(new Net<Dtype>(param));
  }

  shared_ptr<Net<Dtype> > net_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(NetTest, Dtypes);

TYPED_TEST(NetTest, TestHasBlob) {
  EXPECT_TRUE(this->net_->has_blob("data"));
  EXPECT_TRUE(this->net_->has_blob("label"));
  EXPECT_TRUE(this->net_->has_blob("innerproduct"));
  EXPECT_FALSE(this->net_->has_blob("loss"));
  EXPECT_TRUE(this->net_->has_blob("top_loss"));
}

TYPED_TEST(NetTest, TestGetBlob) {
  EXPECT_EQ(this->net_->blob_by_name("data"), this->net_->blobs()[0]);
  EXPECT_EQ(this->net_->blob_by_name("label"), this->net_->blobs()[1]);
  EXPECT_EQ(this->net_->blob_by_name("innerproduct"), this->net_->blobs()[2]);
  EXPECT_FALSE(this->net_->blob_by_name("loss"));
  EXPECT_EQ(this->net_->blob_by_name("top_loss"), this->net_->blobs()[3]);
}

TYPED_TEST(NetTest, TestHasLayer) {
  EXPECT_TRUE(this->net_->has_layer("data"));
  EXPECT_TRUE(this->net_->has_layer("innerproduct"));
  EXPECT_TRUE(this->net_->has_layer("loss"));
  EXPECT_FALSE(this->net_->has_layer("label"));
}

TYPED_TEST(NetTest, TestGetLayerByName) {
  EXPECT_EQ(this->net_->layer_by_name("data"), this->net_->layers()[0]);
  EXPECT_EQ(this->net_->layer_by_name("innerproduct"), this->net_->layers()[1]);
  EXPECT_EQ(this->net_->layer_by_name("loss"), this->net_->layers()[2]);
  EXPECT_FALSE(this->net_->layer_by_name("label"));
}

}  // namespace caffe
