#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/weighted_euclidean_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class WeightedEuclideanLossLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  WeightedEuclideanLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_certainty_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0);
    filler_param.set_max(1);
    GaussianFiller<Dtype> gaussian_filler(filler_param);
    gaussian_filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    gaussian_filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    UniformFiller<Dtype> uniform_filler(filler_param);
    uniform_filler.Fill(this->blob_bottom_certainty_);
    blob_bottom_vec_.push_back(blob_bottom_certainty_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~WeightedEuclideanLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_bottom_certainty_;
    delete blob_top_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifying a weight of 1.
    LayerParameter layer_param;
    WeightedEuclideanLossLayer<Dtype> layer_weight(layer_param);
    layer_weight.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss =
        layer_weight.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype expected_loss = 0;
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      Dtype actual_label = this->blob_bottom_data_->cpu_data()[i];
      Dtype expected_label = this->blob_bottom_label_->cpu_data()[i];
      Dtype weight = this->blob_bottom_certainty_->cpu_data()[i];
      Dtype discrepancy = actual_label - expected_label;
      expected_loss += weight * discrepancy * discrepancy;
    }
    expected_loss /= (this->blob_bottom_data_->num() * Dtype(2));
    EXPECT_NEAR(loss, expected_loss, 1e-6);
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_certainty_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(WeightedEuclideanLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(WeightedEuclideanLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(WeightedEuclideanLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WeightedEuclideanLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 2e-2, 1701, 1, 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
