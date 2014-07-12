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
class SliceLayerTest : public ::testing::Test {
 protected:
  SliceLayerTest()
      : blob_bottom_(new Blob<Dtype>(6, 12, 6, 5)),
        blob_top_0(new Blob<Dtype>()),
        blob_top_1(new Blob<Dtype>()),
        blob_top_2(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    ConstantFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_0.push_back(blob_top_0);
    blob_top_vec_0.push_back(blob_top_1);
    blob_top_vec_1.push_back(blob_top_0);
    blob_top_vec_1.push_back(blob_top_1);
    blob_top_vec_1.push_back(blob_top_2);
    blob_bottom_vec_.push_back(blob_bottom_);
  }

  virtual ~SliceLayerTest() {
    delete blob_top_0; delete blob_top_1;
    delete blob_top_2; delete blob_bottom_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_0;
  Blob<Dtype>* const blob_top_1;
  Blob<Dtype>* const blob_top_2;
  vector<Blob<Dtype>*> blob_top_vec_0, blob_top_vec_1;
  vector<Blob<Dtype>*> blob_bottom_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(SliceLayerTest, Dtypes);

TYPED_TEST(SliceLayerTest, TestCPUSetupNum) {
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->set_slice_dim(0);
  SliceLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_1));
  EXPECT_EQ(this->blob_bottom_->num(),
    this->blob_top_0->num() + this->blob_top_1->num() +
    this->blob_top_2->num());
  EXPECT_EQ(this->blob_bottom_->channels(), this->blob_top_0->channels());
  EXPECT_EQ(this->blob_bottom_->height(), this->blob_top_0->height());
  EXPECT_EQ(this->blob_bottom_->width(), this->blob_top_0->width());
}

TYPED_TEST(SliceLayerTest, TestCPUSetupChannels) {
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->add_slice_point(3);
  SliceLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_0));
  EXPECT_EQ(this->blob_top_0->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_0->channels(), 3);
  EXPECT_EQ(this->blob_top_1->channels(), 9);
  EXPECT_EQ(this->blob_bottom_->channels(),
    this->blob_top_0->channels()+this->blob_top_1->channels());
  EXPECT_EQ(this->blob_bottom_->height(), this->blob_top_0->height());
  EXPECT_EQ(this->blob_bottom_->width(), this->blob_top_0->width());
}


TYPED_TEST(SliceLayerTest, TestCPUNum) {
  LayerParameter layer_param;
  SliceLayer<TypeParam> layer(layer_param);
  Caffe::set_mode(Caffe::CPU);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_0));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_0));
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_top_0->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c, h, w),
            this->blob_top_vec_0[0]->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_top_1->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c+3, h, w),
            this->blob_top_vec_0[1]->data_at(n, c, h, w));
        }
      }
    }
  }
}


TYPED_TEST(SliceLayerTest, TestCPUGradient) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  SliceLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradient(&layer, &(this->blob_bottom_vec_),
    &(this->blob_top_vec_0));
}

TYPED_TEST(SliceLayerTest, TestGPUGradient) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  SliceLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradient(&layer, &(this->blob_bottom_vec_),
    &(this->blob_top_vec_0));
}

}  // namespace caffe
