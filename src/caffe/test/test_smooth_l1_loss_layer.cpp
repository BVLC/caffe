#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/smooth_l1_loss_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
template <typename Dtype>
class SmoothL1LossLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  SmoothL1LossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_inside_weights_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_outside_weights_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter const_filler_param;
    const_filler_param.set_value(-1.);
    ConstantFiller<Dtype> const_filler(const_filler_param);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);

    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);

    filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);

    filler.Fill(this->blob_bottom_inside_weights_);
    blob_bottom_vec_.push_back(blob_bottom_inside_weights_);

    filler.Fill(this->blob_bottom_outside_weights_);
    blob_bottom_vec_.push_back(blob_bottom_outside_weights_);

    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SmoothL1LossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_bottom_inside_weights_;
    delete blob_bottom_outside_weights_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_inside_weights_;
  Blob<Dtype>* const blob_bottom_outside_weights_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SmoothL1LossLayerTest, TestDtypes);

TYPED_TEST(SmoothL1LossLayerTest, TestGradient) {
  LayerParameter layer_param;
  SmoothL1LossParameter* loss_param =
      layer_param.mutable_smooth_l1_loss_param();
  loss_param->set_sigma(2.4);

  const TypeParam kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  SmoothL1LossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}
#endif

}  // namespace caffe
