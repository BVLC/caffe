// Copyright 2014 BVLC and contributors.

#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class PowerLayerTest : public ::testing::Test {
 protected:
  PowerLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PowerLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(PowerLayerTest, Dtypes);

TYPED_TEST(PowerLayerTest, TestPowerCPU) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  TypeParam power = 0.37;
  TypeParam scale = 0.83;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    TypeParam expected_value = pow(shift + scale * bottom_data[i], power);
    if (isnan(expected_value)) {
      EXPECT_TRUE(isnan(top_data[i]));
    } else {
      TypeParam precision = expected_value * 0.0001;
      precision *= (precision < 0) ? -1 : 1;
      EXPECT_NEAR(expected_value, top_data[i], precision);
    }
  }
}

TYPED_TEST(PowerLayerTest, TestPowerGradientCPU) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  TypeParam power = 0.37;
  TypeParam scale = 0.83;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  // Avoid NaNs by forcing (shift + scale * x) >= 0
  TypeParam* bottom_data = this->blob_bottom_->mutable_cpu_data();
  TypeParam min_value = -shift / scale;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (bottom_data[i] < min_value) {
      bottom_data[i] = min_value + (min_value - bottom_data[i]);
    }
  }
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701, 0., 0.01);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(PowerLayerTest, TestPowerGradientShiftZeroCPU) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  TypeParam power = 0.37;
  TypeParam scale = 0.83;
  TypeParam shift = 0.0;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  // Flip negative values in bottom vector as x < 0 -> x^0.37 = nan
  TypeParam* bottom_data = this->blob_bottom_->mutable_cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    bottom_data[i] *= (bottom_data[i] < 0) ? -1 : 1;
  }
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701, 0., 0.01);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(PowerLayerTest, TestPowerZeroCPU) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  TypeParam power = 0.0;
  TypeParam scale = 0.83;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  TypeParam expected_value = TypeParam(1);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(expected_value, top_data[i]);
  }
}

TYPED_TEST(PowerLayerTest, TestPowerZeroGradientCPU) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  TypeParam power = 0.0;
  TypeParam scale = 0.83;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701, 0., 0.01);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(PowerLayerTest, TestPowerOneCPU) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  TypeParam power = 1.0;
  TypeParam scale = 0.83;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    TypeParam expected_value = shift + scale * bottom_data[i];
    EXPECT_NEAR(expected_value, top_data[i], 0.001);
  }
}

TYPED_TEST(PowerLayerTest, TestPowerOneGradientCPU) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  TypeParam power = 1.0;
  TypeParam scale = 0.83;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701, 0., 0.01);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(PowerLayerTest, TestPowerTwoCPU) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  TypeParam power = 2.0;
  TypeParam scale = 0.34;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    TypeParam expected_value = pow(shift + scale * bottom_data[i], 2);
    EXPECT_NEAR(expected_value, top_data[i], 0.001);
  }
}

TYPED_TEST(PowerLayerTest, TestPowerTwoGradientCPU) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  TypeParam power = 2.0;
  TypeParam scale = 0.83;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701, 0., 0.01);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(PowerLayerTest, TestPowerTwoScaleHalfGradientCPU) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  TypeParam power = 2.0;
  TypeParam scale = 0.5;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701, 0., 0.01);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(PowerLayerTest, TestPowerGPU) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  TypeParam power = 0.37;
  TypeParam scale = 0.83;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    TypeParam expected_value = pow(shift + scale * bottom_data[i], power);
    if (isnan(expected_value)) {
      EXPECT_TRUE(isnan(top_data[i]));
    } else {
      TypeParam precision = expected_value * 0.0001;
      precision *= (precision < 0) ? -1 : 1;
      EXPECT_NEAR(expected_value, top_data[i], precision);
    }
  }
}

TYPED_TEST(PowerLayerTest, TestPowerGradientGPU) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  TypeParam power = 0.37;
  TypeParam scale = 0.83;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  // Avoid NaNs by forcing (shift + scale * x) >= 0
  TypeParam* bottom_data = this->blob_bottom_->mutable_cpu_data();
  TypeParam min_value = -shift / scale;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (bottom_data[i] < min_value) {
      bottom_data[i] = min_value + (min_value - bottom_data[i]);
    }
  }
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701, 0., 0.01);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(PowerLayerTest, TestPowerGradientShiftZeroGPU) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  TypeParam power = 0.37;
  TypeParam scale = 0.83;
  TypeParam shift = 0.0;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  // Flip negative values in bottom vector as x < 0 -> x^0.37 = nan
  TypeParam* bottom_data = this->blob_bottom_->mutable_cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    bottom_data[i] *= (bottom_data[i] < 0) ? -1 : 1;
  }
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701, 0., 0.01);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(PowerLayerTest, TestPowerZeroGPU) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  TypeParam power = 0.0;
  TypeParam scale = 0.83;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  TypeParam expected_value = TypeParam(1);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(expected_value, top_data[i]);
  }
}

TYPED_TEST(PowerLayerTest, TestPowerZeroGradientGPU) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  TypeParam power = 0.0;
  TypeParam scale = 0.83;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701, 0., 0.01);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(PowerLayerTest, TestPowerOneGPU) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  TypeParam power = 1.0;
  TypeParam scale = 0.83;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    TypeParam expected_value = shift + scale * bottom_data[i];
    EXPECT_NEAR(expected_value, top_data[i], 0.001);
  }
}

TYPED_TEST(PowerLayerTest, TestPowerOneGradientGPU) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  TypeParam power = 1.0;
  TypeParam scale = 0.83;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701, 0., 0.01);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(PowerLayerTest, TestPowerTwoGPU) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  TypeParam power = 2.0;
  TypeParam scale = 0.34;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    TypeParam expected_value = pow(shift + scale * bottom_data[i], 2);
    EXPECT_NEAR(expected_value, top_data[i], 0.001);
  }
}

TYPED_TEST(PowerLayerTest, TestPowerTwoGradientGPU) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  TypeParam power = 2.0;
  TypeParam scale = 0.83;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701, 0., 0.01);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(PowerLayerTest, TestPowerTwoScaleHalfGradientGPU) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  TypeParam power = 2.0;
  TypeParam scale = 0.5;
  TypeParam shift = -2.4;
  layer_param.mutable_power_param()->set_power(power);
  layer_param.mutable_power_param()->set_scale(scale);
  layer_param.mutable_power_param()->set_shift(shift);
  PowerLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701, 0., 0.01);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

}  // namespace caffe
