#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/loss_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#define NUM 100
#define DIM 15

template <typename TypeParam>
class WeightedLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  WeightedLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(NUM, DIM, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(NUM, 1, 1, 1)),
        blob_bottom_weights_(new Blob<Dtype>(1, 1, DIM, DIM)),
        blob_top_loss_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    PositiveUnitballFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % DIM;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    filler_param.set_min(0.1);
    filler_param.set_max(20.0);
    UniformFiller<Dtype> weighted_filler(filler_param);
    weighted_filler.Fill(this->blob_bottom_weights_);
    blob_bottom_vec_.push_back(blob_bottom_weights_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~WeightedLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_bottom_weights_;
    delete blob_top_loss_;
  }
  void TestForward() {
    LayerParameter layer_param;
    WeightedLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // check that we have an actuall loss at "top"
    EXPECT_EQ(this->blob_top_loss_->count(), 1);
    const Dtype computed_loss = this->blob_top_loss_->cpu_data()[0];

    // compute the loss manually
    const int num = this->blob_bottom_data_->num();
    const int dim = this->blob_bottom_data_->channels();
    EXPECT_EQ(num, NUM);
    EXPECT_EQ(dim, DIM);
    Dtype loss = 0;
    for (int i = 0 ; i < num ; i++) {
      int label = static_cast<int>(this->blob_bottom_label_->cpu_data()[i]);
      for (int j = 0 ; j < dim ; j++) {
        loss += this->blob_bottom_weights_->cpu_data()[label*dim + j] *
            this->blob_bottom_data_->cpu_data()[i*DIM + j];
      }
    }
    loss /= num;
    EXPECT_NEAR(loss, computed_loss, 1e-5);
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_weights_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(WeightedLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(WeightedLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WeightedLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 2e-2, 1701, -1, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WeightedLossLayerTest, TestForward) {
  this->TestForward();
}


}  // namespace caffe
