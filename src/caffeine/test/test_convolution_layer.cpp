#include <cstring>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffeine/blob.hpp"
#include "caffeine/common.hpp"
#include "caffeine/filler.hpp"
#include "caffeine/vision_layers.hpp"
#include "caffeine/test/test_gradient_check_util.hpp"

#include "caffeine/test/test_caffeine_main.hpp"

namespace caffeine {
 
extern cudaDeviceProp CAFFEINE_TEST_CUDA_PROP;

template <typename Dtype>
class ConvolutionLayerTest : public ::testing::Test {
 protected:
  ConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {};
  virtual void SetUp() {
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  };

  virtual ~ConvolutionLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(ConvolutionLayerTest, Dtypes);

TYPED_TEST(ConvolutionLayerTest, TestSetup) {
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_num_output(4);
  shared_ptr<Layer<TypeParam> > layer(
      new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 2);
  // setting group should not change the shape
  layer_param.set_num_output(3);
  layer_param.set_group(3);
  layer.reset(new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 2);
}

TYPED_TEST(ConvolutionLayerTest, TestSimpleConvolution) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_num_output(4);
  layer_param.mutable_weight_filler()->set_type("constant");
  layer_param.mutable_weight_filler()->set_value(1);
  layer_param.mutable_bias_filler()->set_type("constant");
  layer_param.mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > layer(
      new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffeine::set_mode(Caffeine::CPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 27.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_GE(top_data[i], 27.1 - 1e-4);
    EXPECT_LE(top_data[i], 27.1 + 1e-4);
  }
  // Test GPU
  Caffeine::set_mode(Caffeine::GPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 27.1
  top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_GE(top_data[i], 27.1 - 1e-4);
    EXPECT_LE(top_data[i], 27.1 + 1e-4);
  }
}

TYPED_TEST(ConvolutionLayerTest, TestSimpleConvolutionGroup) {
  // We will simply see if the convolution layer carries out averaging well.
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_num_output(3);
  layer_param.set_group(3);
  layer_param.mutable_weight_filler()->set_type("constant");
  layer_param.mutable_weight_filler()->set_value(1);
  layer_param.mutable_bias_filler()->set_type("constant");
  layer_param.mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > layer(
      new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Caffeine::set_mode(Caffeine::CPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 9.1
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_GE(top_data[i], 9.1 - 1e-4);
    EXPECT_LE(top_data[i], 9.1 + 1e-4);
  }
  // Test GPU
  Caffeine::set_mode(Caffeine::GPU);
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the convolution, the output should all have output values 9.1
  top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_GE(top_data[i], 9.1 - 1e-4);
    EXPECT_LE(top_data[i], 9.1 + 1e-4);
  }
}


TYPED_TEST(ConvolutionLayerTest, TestCPUGradient) {
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_num_output(2);
  Caffeine::set_mode(Caffeine::CPU);
  ConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(ConvolutionLayerTest, TestGPUGradient) {
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_num_output(2);
  Caffeine::set_mode(Caffeine::GPU);
  ConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

}
