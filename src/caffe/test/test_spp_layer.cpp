#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/spp_layer.hpp"


#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SPPLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SPPLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_bottom_2_(new Blob<Dtype>()),
        blob_bottom_3_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 9, 8);
    blob_bottom_2_->Reshape(4, 3, 1024, 765);
    blob_bottom_3_->Reshape(10, 3, 7, 7);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_2_.push_back(blob_bottom_2_);
    blob_bottom_vec_3_.push_back(blob_bottom_3_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SPPLayerTest() { delete blob_bottom_; delete blob_top_; }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_bottom_3_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_2_;
  vector<Blob<Dtype>*> blob_bottom_vec_3_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SPPLayerTest, TestDtypesAndDevices);

TYPED_TEST(SPPLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_spp_param()->set_pyramid_height(3);
  SPPLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // expected number of pool results is geometric sum
  // (1 - r ** n)/(1 - r) where r = 4 and n = pyramid_height
  // (1 - 4 ** 3)/(1 - 4) = 21
  // multiply bottom num_channels * expected_pool_results
  // to get expected num_channels (3 * 21 = 63)
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 63);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(SPPLayerTest, TestEqualOutputDims) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_spp_param()->set_pyramid_height(5);
  SPPLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_2_, this->blob_top_vec_);
  // expected number of pool results is geometric sum
  // (1 - r ** n)/(1 - r) where r = 4 and n = pyramid_height
  // (1 - 4 ** 5)/(1 - 4) = 341
  // multiply bottom num_channels * expected_pool_results
  // to get expected num_channels (3 * 341 = 1023)
  EXPECT_EQ(this->blob_top_->num(), 4);
  EXPECT_EQ(this->blob_top_->channels(), 1023);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(SPPLayerTest, TestEqualOutputDims2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_spp_param()->set_pyramid_height(3);
  SPPLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_3_, this->blob_top_vec_);
  // expected number of pool results is geometric sum
  // (1 - r ** n)/(1 - r) where r = 4 and n = pyramid_height
  // (1 - 4 ** 3)/(1 - 4) = 21
  // multiply bottom num_channels * expected_pool_results
  // to get expected num_channels (3 * 21 = 63)
  EXPECT_EQ(this->blob_top_->num(), 10);
  EXPECT_EQ(this->blob_top_->channels(), 63);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(SPPLayerTest, TestForwardBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_spp_param()->set_pyramid_height(3);
  SPPLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);
}

TYPED_TEST(SPPLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SPPParameter* spp_param = layer_param.mutable_spp_param();
  spp_param->set_pyramid_height(3);
  SPPLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


}  // namespace caffe
