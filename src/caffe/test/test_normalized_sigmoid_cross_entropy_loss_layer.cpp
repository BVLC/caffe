#include <cmath>
#include <cstdlib>
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
class NormalizedSigmoidCrossEntropyLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  NormalizedSigmoidCrossEntropyLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_targets_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // Fill the data vector
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    data_filler.Fill(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    // Fill the targets vector
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(0);
    targets_filler_param.set_max(1);
    UniformFiller<Dtype> targets_filler(targets_filler_param);
    targets_filler.Fill(blob_bottom_targets_);
    blob_bottom_vec_.push_back(blob_bottom_targets_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~NormalizedSigmoidCrossEntropyLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_targets_;
    delete blob_top_loss_;
  }

  Dtype NormalizedSigmoidCrossEntropyLossReference(const int count, const int num,
                                         const Dtype* input,
                                         const Dtype* target) {
    Dtype loss = 0;
    const int dim = count / num;
    for (int i = 0; i < dim; ++i) {
      int n_pos = 0;
      int n_neg = 0;
      Dtype pos_loss = 0;
      Dtype neg_loss = 0;
      for (int j = 0; j < num; ++j) {
        const int idx = j * dim + i;
        const Dtype prediction = 1 / (1 + exp(-input[idx]));
        EXPECT_LE(prediction, 1);
        EXPECT_GE(prediction, 0);
        EXPECT_LE(target[idx], 1);
        EXPECT_GE(target[idx], 0);
        if (target[idx] >= 0.5) {
          n_pos++;
          pos_loss -= target[idx] * log(prediction + (target[idx] == Dtype(0)));
          pos_loss -= (1 - target[idx]) * log(1 - prediction + (target[idx] == Dtype(1)));
        } else {
          n_neg++;
          neg_loss -= target[idx] * log(prediction + (target[idx] == Dtype(0)));
          neg_loss -= (1 - target[idx]) * log(1 - prediction + (target[idx] == Dtype(1)));
        }
      }
      if (n_pos > 0) {
        loss += pos_loss / n_pos;
      }
      if (n_neg > 0) {
        loss += neg_loss / n_neg;
      }
    }
    return loss;
  }

  void TestForward() {
    LayerParameter layer_param;
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(0.0);
    targets_filler_param.set_max(1.0);
    UniformFiller<Dtype> targets_filler(targets_filler_param);
    Dtype eps = 2e-2;
    for (int i = 0; i < 100; ++i) {
      // Fill the data vector
      data_filler.Fill(this->blob_bottom_data_);
      // Fill the targets vector
      targets_filler.Fill(this->blob_bottom_targets_);
      NormalizedSigmoidCrossEntropyLossLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
      Dtype layer_loss =
          layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
      const int count = this->blob_bottom_data_->count();
      const int num = this->blob_bottom_data_->num();
      const Dtype* blob_bottom_data = this->blob_bottom_data_->cpu_data();
      const Dtype* blob_bottom_targets =
          this->blob_bottom_targets_->cpu_data();
      Dtype reference_loss = kLossWeight * NormalizedSigmoidCrossEntropyLossReference(
          count, num, blob_bottom_data, blob_bottom_targets);
      EXPECT_NEAR(reference_loss, layer_loss, eps) << "debug: trial #" << i;
    }
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_targets_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(NormalizedSigmoidCrossEntropyLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(NormalizedSigmoidCrossEntropyLossLayerTest, TestNormalizedSigmoidCrossEntropyLoss) {
  this->TestForward();
}

TYPED_TEST(NormalizedSigmoidCrossEntropyLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  NormalizedSigmoidCrossEntropyLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0);
}


}  // namespace caffe
