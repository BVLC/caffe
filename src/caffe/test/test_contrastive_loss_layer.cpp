#include <algorithm>
#include <cmath>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
#include <cstdlib>
#include <cstring>
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> device-abstraction
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
#include "caffe/loss_layers.hpp"
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
#include "caffe/loss_layers.hpp"
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
#include "caffe/loss_layers.hpp"
=======
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/loss_layers.hpp"
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/loss_layers.hpp"
=======
>>>>>>> pod/caffe-merge
=======
#include "caffe/loss_layers.hpp"
=======
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/loss_layers.hpp"
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/loss_layers.hpp"
=======
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/loss_layers.hpp"
=======
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/loss_layers.hpp"
=======
>>>>>>> pod/device/blob.hpp
#include "caffe/vision_layers.hpp"
>>>>>>> origin/BVLC/parallel
=======
#include "caffe/loss_layers.hpp"
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
#include "caffe/loss_layers.hpp"
>>>>>>> device-abstraction
=======
#include "caffe/layers/contrastive_loss_layer.hpp"
>>>>>>> BVLC/master
=======
#include "caffe/layers/contrastive_loss_layer.hpp"
>>>>>>> BVLC/master
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/loss_layers.hpp"
>>>>>>> device-abstraction

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ContrastiveLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ContrastiveLossLayerTest()
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
      : blob_bottom_data_i_(new Blob<Dtype>(512, 2, 1, 1)),
        blob_bottom_data_j_(new Blob<Dtype>(512, 2, 1, 1)),
        blob_bottom_y_(new Blob<Dtype>(512, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);  // distances~=1.0 to test both sides of margin
    UniformFiller<Dtype> filler(filler_param);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
      : blob_bottom_data_i_(new Blob<Dtype>(128, 10, 1, 1)),
        blob_bottom_data_j_(new Blob<Dtype>(128, 10, 1, 1)),
        blob_bottom_y_(new Blob<Dtype>(128, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(0.3);  // distances~=1.0 to test both sides of margin
    GaussianFiller<Dtype> filler(filler_param);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> device-abstraction
    filler.Fill(this->blob_bottom_data_i_);
    blob_bottom_vec_.push_back(blob_bottom_data_i_);
    filler.Fill(this->blob_bottom_data_j_);
    blob_bottom_vec_.push_back(blob_bottom_data_j_);
    for (int i = 0; i < blob_bottom_y_->count(); ++i) {
      blob_bottom_y_->mutable_cpu_data()[i] = caffe_rng_rand() % 2;  // 0 or 1
    }
    blob_bottom_vec_.push_back(blob_bottom_y_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~ContrastiveLossLayerTest() {
    delete blob_bottom_data_i_;
    delete blob_bottom_data_j_;
    delete blob_bottom_y_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_data_i_;
  Blob<Dtype>* const blob_bottom_data_j_;
  Blob<Dtype>* const blob_bottom_y_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ContrastiveLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(ContrastiveLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ContrastiveLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  const Dtype margin = layer_param.contrastive_loss_param().margin();
  const int num = this->blob_bottom_data_i_->num();
  const int channels = this->blob_bottom_data_i_->channels();
  Dtype loss(0);
  for (int i = 0; i < num; ++i) {
    Dtype dist_sq(0);
    for (int j = 0; j < channels; ++j) {
      Dtype diff = this->blob_bottom_data_i_->cpu_data()[i*channels+j] -
          this->blob_bottom_data_j_->cpu_data()[i*channels+j];
      dist_sq += diff*diff;
    }
    if (this->blob_bottom_y_->cpu_data()[i]) {  // similar pairs
      loss += dist_sq;
    } else {
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
      Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq), 0.0);
      loss += dist*dist;
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
      Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq), 0.0);
      loss += dist*dist;
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
      Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq), 0.0);
      loss += dist*dist;
=======
>>>>>>> pod/device/blob.hpp
=======
      Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq), 0.0);
      loss += dist*dist;
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
      Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq), 0.0);
      loss += dist*dist;
=======
>>>>>>> pod/caffe-merge
      loss += std::max(margin-dist_sq, Dtype(0));
>>>>>>> origin/BVLC/parallel
=======
      Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq), 0.0);
      loss += dist*dist;
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
      Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq), 0.0);
      loss += dist*dist;
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
      Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq), 0.0);
      loss += dist*dist;
=======
      loss += std::max(margin-dist_sq, Dtype(0));
>>>>>>> origin/BVLC/parallel
=======
      Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq), 0.0);
      loss += dist*dist;
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
      Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq), 0.0);
      loss += dist*dist;
>>>>>>> device-abstraction
    }
  }
  loss /= static_cast<Dtype>(num) * Dtype(2);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6);
}

TYPED_TEST(ContrastiveLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ContrastiveLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  // check the gradient for the first two bottom layers
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
TYPED_TEST(ContrastiveLossLayerTest, TestForwardLegacy) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_contrastive_loss_param()->set_legacy_version(true);
  ContrastiveLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  const Dtype margin = layer_param.contrastive_loss_param().margin();
  const int num = this->blob_bottom_data_i_->num();
  const int channels = this->blob_bottom_data_i_->channels();
  Dtype loss(0);
  for (int i = 0; i < num; ++i) {
    Dtype dist_sq(0);
    for (int j = 0; j < channels; ++j) {
      Dtype diff = this->blob_bottom_data_i_->cpu_data()[i*channels+j] -
          this->blob_bottom_data_j_->cpu_data()[i*channels+j];
      dist_sq += diff*diff;
    }
    if (this->blob_bottom_y_->cpu_data()[i]) {  // similar pairs
      loss += dist_sq;
    } else {
      loss += std::max(margin - dist_sq, Dtype(0.0));
    }
  }
  loss /= static_cast<Dtype>(num) * Dtype(2);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6);
}

TYPED_TEST(ContrastiveLossLayerTest, TestGradientLegacy) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_contrastive_loss_param()->set_legacy_version(true);
  ContrastiveLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  // check the gradient for the first two bottom layers
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> device-abstraction
}  // namespace caffe
