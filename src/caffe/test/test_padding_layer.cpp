#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class PaddingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  PaddingLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PaddingLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PaddingLayerTest, TestDtypesAndDevices);

TYPED_TEST(PaddingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PaddingParameter* padding_param =
      layer_param.mutable_padding_param();
  padding_param->set_pad_beg(2);
  padding_param->set_pad_end(1);
  PaddingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 9);
  EXPECT_EQ(this->blob_top_->width(), 8);
}

TYPED_TEST(PaddingLayerTest, TestGradientPosPad) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PaddingParameter* padding_param =
      layer_param.mutable_padding_param();
  padding_param->set_pad_beg(2);
  padding_param->set_pad_end(1);
  PaddingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(PaddingLayerTest, TestGradientNegPad) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PaddingParameter* padding_param =
      layer_param.mutable_padding_param();
  padding_param->set_pad_beg(-1);
  padding_param->set_pad_end(-2);
  PaddingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
