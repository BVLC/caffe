#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ReshapeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  ReshapeLayerTest()
    : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
      blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ReshapeLayerTest() { delete blob_bottom_; delete blob_top_; }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ReshapeLayerTest, TestDtypesAndDevices);

TYPED_TEST(ReshapeLayerTest, TestFlattenOutputSizes) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReshapeParameter* reshape_param =
      layer_param.mutable_reshape_param();
  reshape_param->set_channels(-1);
  reshape_param->set_height(1);
  reshape_param->set_width(1);

  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3 * 6 * 5);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(ReshapeLayerTest, TestFlattenValues) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReshapeParameter* reshape_param =
      layer_param.mutable_reshape_param();
  reshape_param->set_channels(-1);
  reshape_param->set_height(1);
  reshape_param->set_width(1);
  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int c = 0; c < 3 * 6 * 5; ++c) {
    EXPECT_EQ(this->blob_top_->data_at(0, c, 0, 0),
        this->blob_bottom_->data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
    EXPECT_EQ(this->blob_top_->data_at(1, c, 0, 0),
        this->blob_bottom_->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
  }
}

// Test whether setting output dimensions to 0 either explicitly or implicitly
// copies the respective dimension of the input layer.
TYPED_TEST(ReshapeLayerTest, TestCopyDimensions) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReshapeParameter* reshape_param =
      layer_param.mutable_reshape_param();
  // Omitting num to test implicit zeroes.
  reshape_param->set_channels(0);
  reshape_param->set_height(0);
  reshape_param->set_width(0);
  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

// When a dimension is set to -1, we should infer its value from the other
// dimensions (including those that get copied from below).
TYPED_TEST(ReshapeLayerTest, TestInferenceOfUnspecified) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReshapeParameter* reshape_param =
      layer_param.mutable_reshape_param();
  // Since omitted, num is implicitly set to 0 (thus, copies 2).
  reshape_param->set_channels(3);
  reshape_param->set_height(10);
  reshape_param->set_width(-1);

  // Count is 180, thus height should be 180 / (2*3*10) = 3.

  ReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

}  // namespace caffe
