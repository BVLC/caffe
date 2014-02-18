// Copyright 2014 Jeff Donahue

#include <cstring>
#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/util/insert_splits.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class SplitLayerTest : public ::testing::Test {
 protected:
  SplitLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_a_(new Blob<Dtype>()),
        blob_top_b_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_a_);
    blob_top_vec_.push_back(blob_top_b_);
  };
  virtual ~SplitLayerTest() {
    delete blob_bottom_;
    delete blob_top_a_;
    delete blob_top_b_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_a_;
  Blob<Dtype>* const blob_top_b_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(SplitLayerTest, Dtypes);

TYPED_TEST(SplitLayerTest, TestSetup) {
  LayerParameter layer_param;
  SplitLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_a_->num(), 2);
  EXPECT_EQ(this->blob_top_a_->channels(), 3);
  EXPECT_EQ(this->blob_top_a_->height(), 6);
  EXPECT_EQ(this->blob_top_a_->width(), 5);
  EXPECT_EQ(this->blob_top_b_->num(), 2);
  EXPECT_EQ(this->blob_top_b_->channels(), 3);
  EXPECT_EQ(this->blob_top_b_->height(), 6);
  EXPECT_EQ(this->blob_top_b_->width(), 5);
}

TYPED_TEST(SplitLayerTest, TestCPU) {
  LayerParameter layer_param;
  SplitLayer<TypeParam> layer(layer_param);
  Caffe::set_mode(Caffe::CPU);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    TypeParam bottom_value = this->blob_bottom_->cpu_data()[i];
    EXPECT_EQ(bottom_value, this->blob_top_a_->cpu_data()[i]);
    EXPECT_EQ(bottom_value, this->blob_top_b_->cpu_data()[i]);
  }
}

TYPED_TEST(SplitLayerTest, TestGPU) {
  LayerParameter layer_param;
  SplitLayer<TypeParam> layer(layer_param);
  Caffe::set_mode(Caffe::GPU);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    TypeParam bottom_value = this->blob_bottom_->cpu_data()[i];
    EXPECT_EQ(bottom_value, this->blob_top_a_->cpu_data()[i]);
    EXPECT_EQ(bottom_value, this->blob_top_b_->cpu_data()[i]);
  }
}

TYPED_TEST(SplitLayerTest, TestCPUInPlace) {
  LayerParameter layer_param;
  SplitLayer<TypeParam> layer(layer_param);
  Caffe::set_mode(Caffe::CPU);
  this->blob_top_vec_[0] = this->blob_bottom_vec_[0];
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    TypeParam bottom_value = this->blob_bottom_->cpu_data()[i];
    EXPECT_EQ(bottom_value, this->blob_top_b_->cpu_data()[i]);
  }
}

TYPED_TEST(SplitLayerTest, TestGPUInPlace) {
  LayerParameter layer_param;
  SplitLayer<TypeParam> layer(layer_param);
  Caffe::set_mode(Caffe::GPU);
  this->blob_top_vec_[0] = this->blob_bottom_vec_[0];
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    TypeParam bottom_value = this->blob_bottom_->cpu_data()[i];
    EXPECT_EQ(bottom_value, this->blob_top_b_->cpu_data()[i]);
  }
}

TYPED_TEST(SplitLayerTest, TestCPUGradient) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  SplitLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SplitLayerTest, TestGPUGradient) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  SplitLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SplitLayerTest, TestCPUGradientInPlace) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  SplitLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  this->blob_top_vec_[0] = this->blob_bottom_vec_[0];
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SplitLayerTest, TestGPUGradientInPlace) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  SplitLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  this->blob_top_vec_[0] = this->blob_bottom_vec_[0];
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


template <typename Dtype>
class SplitLayerInsertionTest : public ::testing::Test {
 protected:
 SplitLayerInsertionTest() { };
  void RunInsertionTest(
      const string& input_param_string, const string& output_param_string) {
    NetParameter input_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        input_param_string, &input_param));
    NetParameter expected_output_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        output_param_string, &expected_output_param));
    NetParameter actual_output_param;
    insert_splits(input_param, &actual_output_param);
    EXPECT_EQ(expected_output_param.DebugString(),
        actual_output_param.DebugString());
  }
};

typedef ::testing::Types<float> InsertionDtypes;
TYPED_TEST_CASE(SplitLayerInsertionTest, InsertionDtypes);

TYPED_TEST(SplitLayerInsertionTest, TestNoInsertion1) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  layer { "
      "    name: 'data' "
      "    type: 'data' "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss' "
      "    type: 'softmax_with_loss' "
      "  } "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunInsertionTest(input_proto, input_proto);
}

TYPED_TEST(SplitLayerInsertionTest, TestNoInsertion2) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  layer { "
      "    name: 'data' "
      "    type: 'data' "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'data_split' "
      "    type: 'split' "
      "  } "
      "  bottom: 'data' "
      "  top: 'data_split_0' "
      "  top: 'data_split_1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod1' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data_split_0' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod2' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data_split_1' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss' "
      "    type: 'euclidean_loss' "
      "  } "
      "  bottom: 'innerprod1' "
      "  bottom: 'innerprod2' "
      "} ";
  this->RunInsertionTest(input_proto, input_proto);
}

TYPED_TEST(SplitLayerInsertionTest, TestNoInsertionImageNet) {
  const string& input_proto =
      "name: 'CaffeNet' "
      "layers { "
      "  layer { "
      "    name: 'data' "
      "    type: 'data' "
      "    source: '/home/jiayq/Data/ILSVRC12/train-leveldb' "
      "    meanfile: '/home/jiayq/Data/ILSVRC12/image_mean.binaryproto' "
      "    batchsize: 256 "
      "    cropsize: 227 "
      "    mirror: true "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'conv1' "
      "    type: 'conv' "
      "    num_output: 96 "
      "    kernelsize: 11 "
      "    stride: 4 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'data' "
      "  top: 'conv1' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'relu1' "
      "    type: 'relu' "
      "  } "
      "  bottom: 'conv1' "
      "  top: 'conv1' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'pool1' "
      "    type: 'pool' "
      "    pool: MAX "
      "    kernelsize: 3 "
      "    stride: 2 "
      "  } "
      "  bottom: 'conv1' "
      "  top: 'pool1' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'norm1' "
      "    type: 'lrn' "
      "    local_size: 5 "
      "    alpha: 0.0001 "
      "    beta: 0.75 "
      "  } "
      "  bottom: 'pool1' "
      "  top: 'norm1' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'pad2' "
      "    type: 'padding' "
      "    pad: 2 "
      "  } "
      "  bottom: 'norm1' "
      "  top: 'pad2' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'conv2' "
      "    type: 'conv' "
      "    num_output: 256 "
      "    group: 2 "
      "    kernelsize: 5 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 1. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'pad2' "
      "  top: 'conv2' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'relu2' "
      "    type: 'relu' "
      "  } "
      "  bottom: 'conv2' "
      "  top: 'conv2' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'pool2' "
      "    type: 'pool' "
      "    pool: MAX "
      "    kernelsize: 3 "
      "    stride: 2 "
      "  } "
      "  bottom: 'conv2' "
      "  top: 'pool2' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'norm2' "
      "    type: 'lrn' "
      "    local_size: 5 "
      "    alpha: 0.0001 "
      "    beta: 0.75 "
      "  } "
      "  bottom: 'pool2' "
      "  top: 'norm2' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'pad3' "
      "    type: 'padding' "
      "    pad: 1 "
      "  } "
      "  bottom: 'norm2' "
      "  top: 'pad3' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'conv3' "
      "    type: 'conv' "
      "    num_output: 384 "
      "    kernelsize: 3 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'pad3' "
      "  top: 'conv3' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'relu3' "
      "    type: 'relu' "
      "  } "
      "  bottom: 'conv3' "
      "  top: 'conv3' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'pad4' "
      "    type: 'padding' "
      "    pad: 1 "
      "  } "
      "  bottom: 'conv3' "
      "  top: 'pad4' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'conv4' "
      "    type: 'conv' "
      "    num_output: 384 "
      "    group: 2 "
      "    kernelsize: 3 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 1. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'pad4' "
      "  top: 'conv4' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'relu4' "
      "    type: 'relu' "
      "  } "
      "  bottom: 'conv4' "
      "  top: 'conv4' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'pad5' "
      "    type: 'padding' "
      "    pad: 1 "
      "  } "
      "  bottom: 'conv4' "
      "  top: 'pad5' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'conv5' "
      "    type: 'conv' "
      "    num_output: 256 "
      "    group: 2 "
      "    kernelsize: 3 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 1. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'pad5' "
      "  top: 'conv5' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'relu5' "
      "    type: 'relu' "
      "  } "
      "  bottom: 'conv5' "
      "  top: 'conv5' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'pool5' "
      "    type: 'pool' "
      "    kernelsize: 3 "
      "    pool: MAX "
      "    stride: 2 "
      "  } "
      "  bottom: 'conv5' "
      "  top: 'pool5' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'fc6' "
      "    type: 'innerproduct' "
      "    num_output: 4096 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.005 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 1. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'pool5' "
      "  top: 'fc6' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'relu6' "
      "    type: 'relu' "
      "  } "
      "  bottom: 'fc6' "
      "  top: 'fc6' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'drop6' "
      "    type: 'dropout' "
      "    dropout_ratio: 0.5 "
      "  } "
      "  bottom: 'fc6' "
      "  top: 'fc6' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'fc7' "
      "    type: 'innerproduct' "
      "    num_output: 4096 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.005 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 1. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'fc6' "
      "  top: 'fc7' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'relu7' "
      "    type: 'relu' "
      "  } "
      "  bottom: 'fc7' "
      "  top: 'fc7' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'drop7' "
      "    type: 'dropout' "
      "    dropout_ratio: 0.5 "
      "  } "
      "  bottom: 'fc7' "
      "  top: 'fc7' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'fc8' "
      "    type: 'innerproduct' "
      "    num_output: 1000 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0 "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'fc7' "
      "  top: 'fc8' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'loss' "
      "    type: 'softmax_loss' "
      "  } "
      "  bottom: 'fc8' "
      "  bottom: 'label' "
      "} ";
  this->RunInsertionTest(input_proto, input_proto);
}

TYPED_TEST(SplitLayerInsertionTest, TestInsertionWithInPlace) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  layer { "
      "    name: 'data' "
      "    type: 'data' "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'relu' "
      "    type: 'relu' "
      "  } "
      "  bottom: 'innerprod' "
      "  top: 'innerprod' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss' "
      "    type: 'softmax_with_loss' "
      "  } "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunInsertionTest(input_proto, input_proto);
}

TYPED_TEST(SplitLayerInsertionTest, TestInsertion) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  layer { "
      "    name: 'data' "
      "    type: 'data' "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod1' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod2' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod3' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data' "
      "  top: 'innerprod3' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss1' "
      "    type: 'euclidean_loss' "
      "  } "
      "  bottom: 'innerprod1' "
      "  bottom: 'innerprod2' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss2' "
      "    type: 'euclidean_loss' "
      "  } "
      "  bottom: 'innerprod2' "
      "  bottom: 'innerprod3' "
      "} ";
  const string& expected_output_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  layer { "
      "    name: 'data' "
      "    type: 'data' "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'data_data_0_split' "
      "    type: 'split' "
      "  } "
      "  bottom: 'data' "
      "  top: 'data' "
      "  top: 'data_data_0_split_1' "
      "  top: 'data_data_0_split_2' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod1' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod2' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data_data_0_split_1' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod2_innerprod2_0_split' "
      "    type: 'split' "
      "  } "
      "  bottom: 'innerprod2' "
      "  top: 'innerprod2' "
      "  top: 'innerprod2_innerprod2_0_split_1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod3' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data_data_0_split_2' "
      "  top: 'innerprod3' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss1' "
      "    type: 'euclidean_loss' "
      "  } "
      "  bottom: 'innerprod1' "
      "  bottom: 'innerprod2' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss2' "
      "    type: 'euclidean_loss' "
      "  } "
      "  bottom: 'innerprod2_innerprod2_0_split_1' "
      "  bottom: 'innerprod3' "
      "} ";
  this->RunInsertionTest(input_proto, expected_output_proto);
}

TYPED_TEST(SplitLayerInsertionTest, TestInsertionTwoTop) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  layer { "
      "    name: 'data' "
      "    type: 'data' "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod1' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod2' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'label' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod3' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data' "
      "  top: 'innerprod3' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod4' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'label' "
      "  top: 'innerprod4' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss1' "
      "    type: 'euclidean_loss' "
      "  } "
      "  bottom: 'innerprod1' "
      "  bottom: 'innerprod3' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss2' "
      "    type: 'euclidean_loss' "
      "  } "
      "  bottom: 'innerprod2' "
      "  bottom: 'innerprod4' "
      "} ";
  const string& expected_output_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  layer { "
      "    name: 'data' "
      "    type: 'data' "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'data_data_0_split' "
      "    type: 'split' "
      "  } "
      "  bottom: 'data' "
      "  top: 'data' "
      "  top: 'data_data_0_split_1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'label_data_1_split' "
      "    type: 'split' "
      "  } "
      "  bottom: 'label' "
      "  top: 'label' "
      "  top: 'label_data_1_split_1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod1' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod2' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'label' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod3' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data_data_0_split_1' "
      "  top: 'innerprod3' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod4' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'label_data_1_split_1' "
      "  top: 'innerprod4' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss1' "
      "    type: 'euclidean_loss' "
      "  } "
      "  bottom: 'innerprod1' "
      "  bottom: 'innerprod3' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss2' "
      "    type: 'euclidean_loss' "
      "  } "
      "  bottom: 'innerprod2' "
      "  bottom: 'innerprod4' "
      "} ";
  this->RunInsertionTest(input_proto, expected_output_proto);
}

TYPED_TEST(SplitLayerInsertionTest, TestInputInsertion) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "input: 'data' "
      "input_dim: 10 "
      "input_dim: 3 "
      "input_dim: 227 "
      "input_dim: 227 "
      "layers: { "
      "  layer { "
      "    name: 'innerprod1' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod2' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss' "
      "    type: 'euclidean_loss' "
      "  } "
      "  bottom: 'innerprod1' "
      "  bottom: 'innerprod2' "
      "} ";
  const string& expected_output_proto =
      "name: 'TestNetwork' "
      "input: 'data' "
      "input_dim: 10 "
      "input_dim: 3 "
      "input_dim: 227 "
      "input_dim: 227 "
      "layers: { "
      "  layer { "
      "    name: 'data_input_0_split' "
      "    type: 'split' "
      "  } "
      "  bottom: 'data' "
      "  top: 'data' "
      "  top: 'data_input_0_split_1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod1' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod2' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data_input_0_split_1' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss' "
      "    type: 'euclidean_loss' "
      "  } "
      "  bottom: 'innerprod1' "
      "  bottom: 'innerprod2' "
      "} ";
  this->RunInsertionTest(input_proto, expected_output_proto);
}

TYPED_TEST(SplitLayerInsertionTest, TestWithInPlace) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  layer { "
      "    name: 'data' "
      "    type: 'data' "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod1' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'relu1' "
      "    type: 'relu' "
      "  } "
      "  bottom: 'innerprod1' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod2' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'innerprod1' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss1' "
      "    type: 'euclidean_loss' "
      "  } "
      "  bottom: 'innerprod1' "
      "  bottom: 'label' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss2' "
      "    type: 'euclidean_loss' "
      "  } "
      "  bottom: 'innerprod2' "
      "  bottom: 'data' "
      "} ";
  const string& expected_output_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  layer { "
      "    name: 'data' "
      "    type: 'data' "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'data_data_0_split' "
      "    type: 'split' "
      "  } "
      "  bottom: 'data' "
      "  top: 'data' "
      "  top: 'data_data_0_split_1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod1' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'data' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'relu1' "
      "    type: 'relu' "
      "  } "
      "  bottom: 'innerprod1' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod1_relu1_0_split' "
      "    type: 'split' "
      "  } "
      "  bottom: 'innerprod1' "
      "  top: 'innerprod1' "
      "  top: 'innerprod1_relu1_0_split_1' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'innerprod2' "
      "    type: 'inner_product' "
      "  } "
      "  bottom: 'innerprod1' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss1' "
      "    type: 'euclidean_loss' "
      "  } "
      "  bottom: 'innerprod1_relu1_0_split_1' "
      "  bottom: 'label' "
      "} "
      "layers: { "
      "  layer { "
      "    name: 'loss2' "
      "    type: 'euclidean_loss' "
      "  } "
      "  bottom: 'innerprod2' "
      "  bottom: 'data_data_0_split_1' "
      "} ";
  this->RunInsertionTest(input_proto, expected_output_proto);
}

}
