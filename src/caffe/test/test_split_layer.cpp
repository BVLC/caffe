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

TYPED_TEST(SplitLayerTest, TestCPUGradient) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  SplitLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_, this->blob_top_vec_);
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SplitLayerTest, TestGPUGradient) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  SplitLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_, this->blob_top_vec_);
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
    CHECK_EQ(expected_output_param.DebugString(),
        actual_output_param.DebugString());
    EXPECT_EQ(expected_output_param.DebugString(),
        actual_output_param.DebugString());
  }
};

typedef ::testing::Types<float> InsertionDtypes;
TYPED_TEST_CASE(SplitLayerInsertionTest, InsertionDtypes);

TYPED_TEST(SplitLayerInsertionTest, TestNoInsertion1) {
  const string& input_proto =
      "name: \"TestNetwork\" "
      "layers: { "
      "  layer { "
      "    name: \"data\" "
      "    type: \"data\" "
      "  } "
      "  top: \"data\" "
      "  top: \"label\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"innerprod\" "
      "    type: \"inner_product\" "
      "  } "
      "  bottom: \"data\" "
      "  top: \"innerprod\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"loss\" "
      "    type: \"softmax_with_loss\" "
      "  } "
      "  bottom: \"innerprod\" "
      "  bottom: \"label\" "
      "} ";
  this->RunInsertionTest(input_proto, input_proto);
}

TYPED_TEST(SplitLayerInsertionTest, TestNoInsertion2) {
  const string& input_proto =
      "name: \"TestNetwork\" "
      "layers: { "
      "  layer { "
      "    name: \"data\" "
      "    type: \"data\" "
      "  } "
      "  top: \"data\" "
      "  top: \"label\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"data_split\" "
      "    type: \"split\" "
      "  } "
      "  bottom: \"data\" "
      "  top: \"data_split_0\" "
      "  top: \"data_split_1\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"innerprod1\" "
      "    type: \"inner_product\" "
      "  } "
      "  bottom: \"data_split_0\" "
      "  top: \"innerprod1\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"innerprod2\" "
      "    type: \"inner_product\" "
      "  } "
      "  bottom: \"data_split_1\" "
      "  top: \"innerprod2\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"loss\" "
      "    type: \"euclidean_loss\" "
      "  } "
      "  bottom: \"innerprod1\" "
      "  bottom: \"innerprod2\" "
      "} ";
  this->RunInsertionTest(input_proto, input_proto);
}

TYPED_TEST(SplitLayerInsertionTest, TestInsertion) {
  const string& input_proto =
      "name: \"TestNetwork\" "
      "layers: { "
      "  layer { "
      "    name: \"data\" "
      "    type: \"data\" "
      "  } "
      "  top: \"data\" "
      "  top: \"label\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"innerprod1\" "
      "    type: \"inner_product\" "
      "  } "
      "  bottom: \"data\" "
      "  top: \"innerprod1\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"innerprod2\" "
      "    type: \"inner_product\" "
      "  } "
      "  bottom: \"data\" "
      "  top: \"innerprod2\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"innerprod3\" "
      "    type: \"inner_product\" "
      "  } "
      "  bottom: \"data\" "
      "  top: \"innerprod3\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"loss\" "
      "    type: \"euclidean_loss\" "
      "  } "
      "  bottom: \"innerprod1\" "
      "  bottom: \"innerprod2\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"loss\" "
      "    type: \"euclidean_loss\" "
      "  } "
      "  bottom: \"innerprod2\" "
      "  bottom: \"innerprod3\" "
      "} ";
  const string& expected_output_proto =
      "name: \"TestNetwork\" "
      "layers: { "
      "  layer { "
      "    name: \"data\" "
      "    type: \"data\" "
      "  } "
      "  top: \"data\" "
      "  top: \"label\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"data_split\" "
      "    type: \"split\" "
      "  } "
      "  bottom: \"data\" "
      "  top: \"data\" "
      "  top: \"data_split_1\" "
      "  top: \"data_split_2\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"innerprod1\" "
      "    type: \"inner_product\" "
      "  } "
      "  bottom: \"data\" "
      "  top: \"innerprod1\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"innerprod2\" "
      "    type: \"inner_product\" "
      "  } "
      "  bottom: \"data_split_1\" "
      "  top: \"innerprod2\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"innerprod2_split\" "
      "    type: \"split\" "
      "  } "
      "  bottom: \"innerprod2\" "
      "  top: \"innerprod2\" "
      "  top: \"innerprod2_split_1\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"innerprod3\" "
      "    type: \"inner_product\" "
      "  } "
      "  bottom: \"data_split_2\" "
      "  top: \"innerprod3\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"loss\" "
      "    type: \"euclidean_loss\" "
      "  } "
      "  bottom: \"innerprod1\" "
      "  bottom: \"innerprod2\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"loss\" "
      "    type: \"euclidean_loss\" "
      "  } "
      "  bottom: \"innerprod2_split_1\" "
      "  bottom: \"innerprod3\" "
      "} ";
  this->RunInsertionTest(input_proto, expected_output_proto);
}

TYPED_TEST(SplitLayerInsertionTest, TestInputInsertion) {
  const string& input_proto =
      "name: \"TestNetwork\" "
      "input: \"data\" "
      "input_dim: 10 "
      "input_dim: 3 "
      "input_dim: 227 "
      "input_dim: 227 "
      "layers: { "
      "  layer { "
      "    name: \"innerprod1\" "
      "    type: \"inner_product\" "
      "  } "
      "  bottom: \"data\" "
      "  top: \"innerprod1\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"innerprod2\" "
      "    type: \"inner_product\" "
      "  } "
      "  bottom: \"data\" "
      "  top: \"innerprod2\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"loss\" "
      "    type: \"euclidean_loss\" "
      "  } "
      "  bottom: \"innerprod1\" "
      "  bottom: \"innerprod2\" "
      "} ";
  const string& expected_output_proto =
      "name: \"TestNetwork\" "
      "input: \"data\" "
      "input_dim: 10 "
      "input_dim: 3 "
      "input_dim: 227 "
      "input_dim: 227 "
      "layers: { "
      "  layer { "
      "    name: \"data_split\" "
      "    type: \"split\" "
      "  } "
      "  bottom: \"data\" "
      "  top: \"data\" "
      "  top: \"data_split_1\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"innerprod1\" "
      "    type: \"inner_product\" "
      "  } "
      "  bottom: \"data\" "
      "  top: \"innerprod1\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"innerprod2\" "
      "    type: \"inner_product\" "
      "  } "
      "  bottom: \"data_split_1\" "
      "  top: \"innerprod2\" "
      "} "
      "layers: { "
      "  layer { "
      "    name: \"loss\" "
      "    type: \"euclidean_loss\" "
      "  } "
      "  bottom: \"innerprod1\" "
      "  bottom: \"innerprod2\" "
      "} ";
  this->RunInsertionTest(input_proto, expected_output_proto);
}

}
