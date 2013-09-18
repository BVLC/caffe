#include <cstring>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffeine/blob.hpp"
#include "caffeine/common.hpp"
#include "caffeine/filler.hpp"
#include "caffeine/vision_layers.hpp"
#include "caffeine/test/test_gradient_check_util.hpp"


namespace caffeine {

extern cudaDeviceProp CAFFEINE_TEST_CUDA_PROP;
  
template <typename Dtype>
class NeuronLayerTest : public ::testing::Test {
 protected:
  NeuronLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  };
  virtual ~NeuronLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(NeuronLayerTest, Dtypes);

TYPED_TEST(NeuronLayerTest, TestReLUCPU) {
  LayerParameter layer_param;
  Caffeine::set_mode(Caffeine::CPU);
  ReLULayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], 0.);
    EXPECT_TRUE(top_data[i] == 0 || top_data[i] == bottom_data[i]);
  }
}


TYPED_TEST(NeuronLayerTest, TestReLUGradientCPU) {
  LayerParameter layer_param;
  Caffeine::set_mode(Caffeine::CPU);
  ReLULayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradient(layer, this->blob_bottom_vec_, this->blob_top_vec_);
}


TYPED_TEST(NeuronLayerTest, TestReLUGPU) {
  LayerParameter layer_param;
  Caffeine::set_mode(Caffeine::GPU);
  ReLULayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], 0.);
    EXPECT_TRUE(top_data[i] == 0 || top_data[i] == bottom_data[i]);
  }
}


TYPED_TEST(NeuronLayerTest, TestReLUGradientGPU) {
  LayerParameter layer_param;
  Caffeine::set_mode(Caffeine::GPU);
  ReLULayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradient(layer, this->blob_bottom_vec_, this->blob_top_vec_);
}


TYPED_TEST(NeuronLayerTest, TestDropoutCPU) {
  LayerParameter layer_param;
  Caffeine::set_mode(Caffeine::CPU);
  Caffeine::set_phase(Caffeine::TRAIN);
  DropoutLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  float scale = 1. / (1. - layer_param.dropout_ratio());
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (top_data[i] != 0) {
      EXPECT_EQ(top_data[i], bottom_data[i] * scale);
    }
  }
}


TYPED_TEST(NeuronLayerTest, TestDropoutGradientCPU) {
  LayerParameter layer_param;
  Caffeine::set_mode(Caffeine::CPU);
  DropoutLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradient(layer, this->blob_bottom_vec_, this->blob_top_vec_);
}


TYPED_TEST(NeuronLayerTest, TestDropoutCPUTestPhase) {
  LayerParameter layer_param;
  Caffeine::set_mode(Caffeine::CPU);
  Caffeine::set_phase(Caffeine::TEST);
  DropoutLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  float scale = 1. / (1. - layer_param.dropout_ratio());
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (top_data[i] != 0) {
      EXPECT_EQ(top_data[i], bottom_data[i]);
    }
  }
}


TYPED_TEST(NeuronLayerTest, TestDropoutGPU) {
  LayerParameter layer_param;
  Caffeine::set_mode(Caffeine::GPU);
  Caffeine::set_phase(Caffeine::TRAIN);
  DropoutLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  float scale = 1. / (1. - layer_param.dropout_ratio());
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (top_data[i] != 0) {
      EXPECT_EQ(top_data[i], bottom_data[i] * scale);
    }
  }
}


TYPED_TEST(NeuronLayerTest, TestDropoutGradientGPU) {
  if (CAFFEINE_TEST_CUDA_PROP.major >= 2) {
    LayerParameter layer_param;
    Caffeine::set_mode(Caffeine::GPU);
    DropoutLayer<TypeParam> layer(layer_param);
    GradientChecker<TypeParam> checker(1e-2, 1e-3);
    checker.CheckGradient(layer, this->blob_bottom_vec_, this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test to spare my laptop.";
  }
}


TYPED_TEST(NeuronLayerTest, TestDropoutGPUTestPhase) {
  LayerParameter layer_param;
  Caffeine::set_mode(Caffeine::GPU);
  Caffeine::set_phase(Caffeine::TEST);
  DropoutLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  float scale = 1. / (1. - layer_param.dropout_ratio());
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (top_data[i] != 0) {
      EXPECT_EQ(top_data[i], bottom_data[i]);
    }
  }
}

}
