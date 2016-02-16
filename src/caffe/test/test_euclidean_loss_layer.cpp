#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/euclidean_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class EuclideanLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  EuclideanLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~EuclideanLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    EuclideanLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    EuclideanLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(EuclideanLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(EuclideanLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(EuclideanLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  EuclideanLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(EuclideanLossLayerTest, TestForwardIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  // First, compute the loss with ignored labels
  LayerParameter layer_param;
  const Dtype kIgnoreLabelValue = -1;
  layer_param.mutable_loss_param()->set_ignore_label(int(kIgnoreLabelValue));
  this->blob_bottom_label_->mutable_cpu_data()[1] = kIgnoreLabelValue;
  this->blob_bottom_label_->mutable_cpu_data()[5] = kIgnoreLabelValue;
  this->blob_bottom_label_->mutable_cpu_data()[10] = kIgnoreLabelValue;
  EuclideanLossLayer<Dtype> layer_ignore(layer_param);
  layer_ignore.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype loss_ignore =
      layer_ignore.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Then, manually compute the loss for known ignored indices
  int count = this->blob_bottom_label_->count();
  Dtype loss_manual = 0;
  for(int i = 0; i < count; ++i) {
    if (i == 1 || i == 5 || i == 10) {
      continue;
    }
    const Dtype a = this->blob_bottom_label_->mutable_cpu_data()[i];
    const Dtype b = this->blob_bottom_data_->mutable_cpu_data()[i];
    Dtype dot = (a-b)*(a-b);
    loss_manual += dot;
  }
  loss_manual = loss_manual / this->blob_bottom_data_->num() / Dtype(2);
  // Finally, compare the two losses
  EXPECT_NEAR(loss_ignore, loss_manual, 1e-4);
}

TYPED_TEST(EuclideanLossLayerTest, TestGradientIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  // First, compute the diffs without ignored labels
  LayerParameter layer_param;
  const Dtype kIgnoreLabelValue = -1;
  EuclideanLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<bool> propagate_down(2);
  propagate_down[0] = true;
  propagate_down[1] = true;
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  int count = this->blob_bottom_label_->count();
  // Next, sum up the diffs (except for ignored indices and label_value that gets ignored during typecasting)
  Dtype accum_grad_a[2] = {0, 0};
  for(int i = 0; i < count; ++i) {
    const int label_value = static_cast<int>(this->blob_bottom_label_->cpu_data()[i]);
    if (i == 2 || i == 6 || i == 11 || label_value == int(kIgnoreLabelValue)) {
      continue;
    }
    accum_grad_a[0] += this->blob_bottom_data_->mutable_cpu_diff()[i];
    accum_grad_a[1] += this->blob_bottom_label_->mutable_cpu_diff()[i];
  }
  // Compute the diffs with ignored labels and sum them up
  layer_param.mutable_loss_param()->set_ignore_label(int(kIgnoreLabelValue));
  this->blob_bottom_label_->mutable_cpu_data()[2] = kIgnoreLabelValue;
  this->blob_bottom_label_->mutable_cpu_data()[6] = kIgnoreLabelValue;
  this->blob_bottom_label_->mutable_cpu_data()[11] = kIgnoreLabelValue;
  EuclideanLossLayer<Dtype> layer_ignore(layer_param);
  layer_ignore.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer_ignore.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  layer_ignore.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);;
  Dtype accum_grad_b[2] = {0, 0};
  for(int i = 0; i < count; ++i) {
    accum_grad_b[0] += this->blob_bottom_data_->mutable_cpu_diff()[i];
    accum_grad_b[1] += this->blob_bottom_label_->mutable_cpu_diff()[i];
  }
  // Compare the diffs
  EXPECT_NEAR(accum_grad_a[0], accum_grad_b[0], 1e-4);
  EXPECT_NEAR(accum_grad_a[1], accum_grad_b[1], 1e-4);
}

}  // namespace caffe
