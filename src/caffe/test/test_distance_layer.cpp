// Copyright 2014 BVLC and contributors.

#include <cstring>
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
class DistanceLayerTest : public ::testing::Test {
 protected:
	DistanceLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(2, 1, 1, 3)),
        blob_bottom_1_(new Blob<Dtype>(2, 1, 1, 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~DistanceLayerTest() {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(DistanceLayerTest, Dtypes);

TYPED_TEST(DistanceLayerTest, TestSetUp) {
  LayerParameter layer_param;
  DistanceParameter* distance_param =
      layer_param.mutable_distance_param();
  distance_param->set_num_output(2);
  shared_ptr<DistanceLayer<TypeParam> > layer(
      new DistanceLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 2);
}

TYPED_TEST(DistanceLayerTest, TestCPU) {
  LayerParameter layer_param;
  DistanceParameter* distance_param =
      layer_param.mutable_distance_param();
  Caffe::set_mode(Caffe::CPU);
  distance_param->set_num_output(2);
  distance_param->mutable_weight_filler()->set_type("uniform");
  distance_param->mutable_bias_filler()->set_type("uniform");
  distance_param->mutable_bias_filler()->set_min(1);
  distance_param->mutable_bias_filler()->set_max(2);
  shared_ptr<DistanceLayer<TypeParam> > layer(
      new DistanceLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], 1.);
  }
}

TYPED_TEST(DistanceLayerTest, TestGPU) {
  if (sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    LayerParameter layer_param;
    DistanceParameter* distance_param =
        layer_param.mutable_distance_param();
    Caffe::set_mode(Caffe::GPU);
    distance_param->set_num_output(2);
    distance_param->mutable_weight_filler()->set_type("uniform");
    distance_param->mutable_bias_filler()->set_type("uniform");
    distance_param->mutable_bias_filler()->set_min(1);
    distance_param->mutable_bias_filler()->set_max(2);
    shared_ptr<DistanceLayer<TypeParam> > layer(
      new DistanceLayer<TypeParam>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
    layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
    const TypeParam* data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 1.);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(DistanceLayerTest, TestCPUGradient) {
  LayerParameter layer_param;
  DistanceParameter* distance_param =
      layer_param.mutable_distance_param();
  Caffe::set_mode(Caffe::CPU);
  distance_param->set_num_output(2);
  distance_param->mutable_weight_filler()->set_type("gaussian");
  distance_param->mutable_bias_filler()->set_type("gaussian");
  distance_param->mutable_bias_filler()->set_min(1);
  distance_param->mutable_bias_filler()->set_max(2);
  DistanceLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(DistanceLayerTest, TestGPUGradient) {
  if (sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    LayerParameter layer_param;
    DistanceParameter* distance_param =
        layer_param.mutable_distance_param();
    Caffe::set_mode(Caffe::GPU);
    distance_param->set_num_output(2);
    distance_param->mutable_weight_filler()->set_type("gaussian");
    distance_param->mutable_bias_filler()->set_type("gaussian");
    DistanceLayer<TypeParam> layer(layer_param);

    GradientChecker<TypeParam> checker(1e-2, 1e-2);
    checker.CheckGradient(&layer, &(this->blob_bottom_vec_),
        &(this->blob_top_vec_));
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
