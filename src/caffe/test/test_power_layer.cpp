#include <algorithm>
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
class PowerLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PowerLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PowerLayerTest() { delete blob_bottom_; delete blob_top_; }

  void TestForward(Dtype power, Dtype scale, Dtype shift) {
    LayerParameter layer_param;
    layer_param.mutable_power_param()->set_power(power);
    layer_param.mutable_power_param()->set_scale(scale);
    layer_param.mutable_power_param()->set_shift(shift);
    PowerLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const Dtype min_precision = 1e-5;
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
      Dtype expected_value = pow(shift + scale * bottom_data[i], power);
      if (power == Dtype(0) || power == Dtype(1) || power == Dtype(2)) {
        EXPECT_FALSE(isnan(top_data[i]));
      }
      if (isnan(expected_value)) {
        EXPECT_TRUE(isnan(top_data[i]));
      } else {
        Dtype precision = std::max(
          Dtype(std::abs(expected_value * Dtype(1e-4))), min_precision);
        EXPECT_NEAR(expected_value, top_data[i], precision);
      }
    }
  }

  void TestBackward(Dtype power, Dtype scale, Dtype shift) {
    LayerParameter layer_param;
    layer_param.mutable_power_param()->set_power(power);
    layer_param.mutable_power_param()->set_scale(scale);
    layer_param.mutable_power_param()->set_shift(shift);
    PowerLayer<Dtype> layer(layer_param);
    if (power != Dtype(0) && power != Dtype(1) && power != Dtype(2)) {
      // Avoid NaNs by forcing (shift + scale * x) >= 0
      Dtype* bottom_data = this->blob_bottom_->mutable_cpu_data();
      Dtype min_value = -shift / scale;
      for (int i = 0; i < this->blob_bottom_->count(); ++i) {
        if (bottom_data[i] < min_value) {
          bottom_data[i] = min_value + (min_value - bottom_data[i]);
        }
      }
    }
    GradientChecker<Dtype> checker(1e-3, 1e-2, 1701, 0., 0.01);
    checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PowerLayerTest, TestDtypesAndDevices);

TYPED_TEST(PowerLayerTest, TestPower) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 0.37;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerGradient) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 0.37;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestBackward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerGradientShiftZero) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 0.37;
  Dtype scale = 0.83;
  Dtype shift = 0.0;
  this->TestBackward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerZero) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 0.0;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerZeroGradient) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 0.0;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestBackward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerOne) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 1.0;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerOneGradient) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 1.0;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestBackward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerTwo) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 2.0;
  Dtype scale = 0.34;
  Dtype shift = -2.4;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerTwoGradient) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 2.0;
  Dtype scale = 0.83;
  Dtype shift = -2.4;
  this->TestBackward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerTwoScaleHalfGradient) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype power = 2.0;
  Dtype scale = 0.5;
  Dtype shift = -2.4;
  this->TestBackward(power, scale, shift);
}

}  // namespace caffe
