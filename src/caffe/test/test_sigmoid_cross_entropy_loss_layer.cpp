#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SigmoidCrossEntropyLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SigmoidCrossEntropyLossLayerTest()
      : batch_size_(10),
        num_labels_(5),
        blob_bottom_data_(new Blob<Dtype>(batch_size_, num_labels_, 1, 1)),
        blob_bottom_targets_(new Blob<Dtype>(batch_size_, num_labels_, 1, 1)),
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
  virtual ~SigmoidCrossEntropyLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_targets_;
    delete blob_top_loss_;
  }

  Dtype SigmoidCrossEntropyLossReference(const int count, const int num,
                                         const Dtype* input,
                                         const Dtype* target) {
    Dtype loss = 0;
    for (int i = 0; i < count; ++i) {
      if (ignore_label_ != 0 && target[i] == ignore_label_) {
        continue;
      }
      const Dtype prediction = 1 / (1 + exp(-input[i]));
      EXPECT_LE(prediction, 1);
      EXPECT_GE(prediction, 0);
      EXPECT_LE(target[i], 1);
      EXPECT_GE(target[i], 0);
      loss -= target[i] * log(prediction + (target[i] == Dtype(0)));
      loss -= (1 - target[i]) * log(1 - prediction + (target[i] == Dtype(1)));
    }
    return loss / num;
  }

  void TestForward(Dtype ignore_label = 0) {
    ignore_label_ = ignore_label;

    LayerParameter layer_param;
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);

    if (ignore_label_ != 0) {
      layer_param.mutable_loss_param()->set_ignore_label(ignore_label_);
    }
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
      this->fill_targets(&targets_filler, this->blob_bottom_targets_);
      // targets_filler.Fill(this->blob_bottom_targets_);
      SigmoidCrossEntropyLossLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      Dtype layer_loss =
          layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      const int count = this->blob_bottom_data_->count();
      const int num = this->blob_bottom_data_->num();
      const Dtype* blob_bottom_data = this->blob_bottom_data_->cpu_data();
      const Dtype* blob_bottom_targets = this->blob_bottom_targets_->cpu_data();
      Dtype reference_loss =
          kLossWeight * SigmoidCrossEntropyLossReference(
                            count, num, blob_bottom_data, blob_bottom_targets);
      EXPECT_NEAR(reference_loss, layer_loss, eps) << "debug: trial #" << i;
    }
  }

  void TestGradient(Dtype ignore_label = 0) {
    ignore_label_ = ignore_label;

    LayerParameter layer_param;
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);

    if (ignore_label_ != 0) {
      layer_param.mutable_loss_param()->set_ignore_label(ignore_label_);
    }
    SigmoidCrossEntropyLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    fill_targets(NULL, this->blob_bottom_targets_);
    GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                    this->blob_top_vec_, 0);
  }

  void fill_targets(UniformFiller<Dtype>* filler, Blob<Dtype>* const targets) {
    // First use the specified filler
    if (filler != NULL) {
      filler->Fill(targets);
    }

    // Now set the ignore labels.
    if (ignore_label_ != 0) {
      for (int i = 0; i < ignore_indexes_.size(); ++i) {
        CHECK_LT(ignore_indexes_[i], targets->count());
        Dtype* t = targets->mutable_cpu_data();
        t[ignore_indexes_[i]] = ignore_label_;
      }
    }
  }

  int batch_size_;
  int num_labels_;
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_targets_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  // Store a list of indices to ignore.
  vector<int> ignore_indexes_;
  Dtype ignore_label_;
};

TYPED_TEST_CASE(SigmoidCrossEntropyLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SigmoidCrossEntropyLossLayerTest, TestSigmoidCrossEntropyLoss) {
  this->TestForward();
}

TYPED_TEST(SigmoidCrossEntropyLossLayerTest,
           TestForwardWithIgnoreLabelsNoMatch) {
  this->TestForward(-1);
}

TYPED_TEST(SigmoidCrossEntropyLossLayerTest, TestForwardWithIgnoreAll) {
  for (int i = 0; i < this->num_labels_ * this->batch_size_; ++i) {
    this->ignore_indexes_.push_back(i);
  }
  this->TestForward(-1);
}

TYPED_TEST(SigmoidCrossEntropyLossLayerTest, TestForwardIgnoreSome) {
  this->ignore_indexes_.push_back(0);
  this->ignore_indexes_.push_back(1);
  this->ignore_indexes_.push_back(7);
  this->ignore_indexes_.push_back(11);
  this->ignore_indexes_.push_back(34);

  this->TestForward(-1);
}

TYPED_TEST(SigmoidCrossEntropyLossLayerTest, TestGradient) {
  this->TestGradient();
}

TYPED_TEST(SigmoidCrossEntropyLossLayerTest,
           TestGradientWithIgnoreLabelsNoMatch) {
  this->TestGradient(-1);
}

TYPED_TEST(SigmoidCrossEntropyLossLayerTest, TestGradientWithIgnoreAll) {
  for (int i = 0; i < this->num_labels_ * this->batch_size_; ++i) {
    this->ignore_indexes_.push_back(i);
  }
  this->TestGradient(-1);
}

TYPED_TEST(SigmoidCrossEntropyLossLayerTest, TestGradientIgnoreSome) {
  this->ignore_indexes_.push_back(0);
  this->ignore_indexes_.push_back(1);
  this->ignore_indexes_.push_back(7);
  this->ignore_indexes_.push_back(11);
  this->ignore_indexes_.push_back(34);

  this->TestGradient(-1);
}

}  // namespace caffe
