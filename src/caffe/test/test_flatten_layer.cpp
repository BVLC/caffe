#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/flatten_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class FlattenLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  FlattenLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~FlattenLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(FlattenLayerTest, TestDtypesAndDevices);

TYPED_TEST(FlattenLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3 * 6 * 5);
}

TYPED_TEST(FlattenLayerTest, TestSetupWithAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_axis(2);
  FlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 3);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3);
  EXPECT_EQ(this->blob_top_->shape(2), 6 * 5);
}

TYPED_TEST(FlattenLayerTest, TestSetupWithEndAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_end_axis(-2);
  FlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 3);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3 * 6);
  EXPECT_EQ(this->blob_top_->shape(2), 5);
}

TYPED_TEST(FlattenLayerTest, TestSetupWithStartAndEndAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_flatten_param()->set_axis(0);
  layer_param.mutable_flatten_param()->set_end_axis(-2);
  FlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), 2 * 3 * 6);
  EXPECT_EQ(this->blob_top_->shape(1), 5);
}

TYPED_TEST(FlattenLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FlattenLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int c = 0; c < 3 * 6 * 5; ++c) {
    EXPECT_EQ(this->blob_top_->data_at(0, c, 0, 0),
        this->blob_bottom_->data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
    EXPECT_EQ(this->blob_top_->data_at(1, c, 0, 0),
        this->blob_bottom_->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
  }
}

TYPED_TEST(FlattenLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FlattenLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
