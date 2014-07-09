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
class EltwiseLayerTest : public ::testing::Test {
 protected:
  EltwiseLayerTest()
      : blob_bottom_a_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_b_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_c_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_a_);
    filler.Fill(this->blob_bottom_b_);
    filler.Fill(this->blob_bottom_c_);
    blob_bottom_vec_.push_back(blob_bottom_a_);
    blob_bottom_vec_.push_back(blob_bottom_b_);
    blob_bottom_vec_.push_back(blob_bottom_c_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~EltwiseLayerTest() {
    delete blob_bottom_a_;
    delete blob_bottom_b_;
    delete blob_bottom_c_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_a_;
  Blob<Dtype>* const blob_bottom_b_;
  Blob<Dtype>* const blob_bottom_c_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(EltwiseLayerTest, Dtypes);

TYPED_TEST(EltwiseLayerTest, TestSetUp) {
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
  shared_ptr<EltwiseLayer<TypeParam> > layer(
      new EltwiseLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(EltwiseLayerTest, TestProdCPU) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
  shared_ptr<EltwiseLayer<TypeParam> > layer(
      new EltwiseLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const TypeParam* in_data_a = this->blob_bottom_a_->cpu_data();
  const TypeParam* in_data_b = this->blob_bottom_b_->cpu_data();
  const TypeParam* in_data_c = this->blob_bottom_c_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(data[i], in_data_a[i] * in_data_b[i] * in_data_c[i]);
  }
}

TYPED_TEST(EltwiseLayerTest, TestSumCPU) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  shared_ptr<EltwiseLayer<TypeParam> > layer(
      new EltwiseLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const TypeParam* in_data_a = this->blob_bottom_a_->cpu_data();
  const TypeParam* in_data_b = this->blob_bottom_b_->cpu_data();
  const TypeParam* in_data_c = this->blob_bottom_c_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(data[i], in_data_a[i] + in_data_b[i] + in_data_c[i]);
  }
}

TYPED_TEST(EltwiseLayerTest, TestSumCoeffCPU) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  eltwise_param->add_coeff(1);
  eltwise_param->add_coeff(-0.5);
  eltwise_param->add_coeff(2);
  shared_ptr<EltwiseLayer<TypeParam> > layer(
      new EltwiseLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const TypeParam* in_data_a = this->blob_bottom_a_->cpu_data();
  const TypeParam* in_data_b = this->blob_bottom_b_->cpu_data();
  const TypeParam* in_data_c = this->blob_bottom_c_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] - 0.5*in_data_b[i] + 2*in_data_c[i],
        1e-4);
  }
}

TYPED_TEST(EltwiseLayerTest, TestProdGPU) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
  shared_ptr<EltwiseLayer<TypeParam> > layer(
      new EltwiseLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const TypeParam* in_data_a = this->blob_bottom_a_->cpu_data();
  const TypeParam* in_data_b = this->blob_bottom_b_->cpu_data();
  const TypeParam* in_data_c = this->blob_bottom_c_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(data[i], in_data_a[i] * in_data_b[i] * in_data_c[i]);
  }
}

TYPED_TEST(EltwiseLayerTest, TestSumGPU) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  shared_ptr<EltwiseLayer<TypeParam> > layer(
      new EltwiseLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const TypeParam* in_data_a = this->blob_bottom_a_->cpu_data();
  const TypeParam* in_data_b = this->blob_bottom_b_->cpu_data();
  const TypeParam* in_data_c = this->blob_bottom_c_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(data[i], in_data_a[i] + in_data_b[i] + in_data_c[i]);
  }
}

TYPED_TEST(EltwiseLayerTest, TestSumCoeffGPU) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  eltwise_param->add_coeff(1);
  eltwise_param->add_coeff(-0.5);
  eltwise_param->add_coeff(2);
  shared_ptr<EltwiseLayer<TypeParam> > layer(
      new EltwiseLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const TypeParam* in_data_a = this->blob_bottom_a_->cpu_data();
  const TypeParam* in_data_b = this->blob_bottom_b_->cpu_data();
  const TypeParam* in_data_c = this->blob_bottom_c_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] - 0.5*in_data_b[i] + 2*in_data_c[i],
        1e-4);
  }
}

TYPED_TEST(EltwiseLayerTest, TestProdCPUGradient) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
  EltwiseLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(EltwiseLayerTest, TestSumCPUGradient) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  EltwiseLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(EltwiseLayerTest, TestSumCoeffCPUGradient) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  eltwise_param->add_coeff(1);
  eltwise_param->add_coeff(-0.5);
  eltwise_param->add_coeff(2);
  EltwiseLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(EltwiseLayerTest, TestSumGPUGradient) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  EltwiseLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(EltwiseLayerTest, TestSumCoeffGPUGradient) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  eltwise_param->add_coeff(1);
  eltwise_param->add_coeff(-0.5);
  eltwise_param->add_coeff(2);
  EltwiseLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

}  // namespace caffe
