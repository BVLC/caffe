// Copyright 2014 Sergio Guadarrama

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

#include <string>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/test/test_caffe_main.hpp"

using std::string;

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class ImagesLayerTest : public ::testing::Test {
 protected:
  ImagesLayerTest()
      : blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        filename(NULL) {};
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    // Create a Vector of files with labels
    filename = tmpnam(NULL); // get temp name
    std::ofstream outfile(filename, std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename;
    for (int i = 0; i < 5; ++i) {
      outfile << "data/cat.jpg " << i;
    }
    outfile.close();
  };

  virtual ~ImagesLayerTest() { delete blob_top_data_; delete blob_top_label_; }

  char* filename;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(ImagesLayerTest, Dtypes);

TYPED_TEST(ImagesLayerTest, TestRead) {
  LayerParameter param;
  param.set_batchsize(5);
  param.set_source(this->filename);
  param.set_shuffle_images(false);
  ImagesLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 1200);
  EXPECT_EQ(this->blob_top_data_->width(), 1600);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data 5 times
  for (int iter = 0; iter < 5; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(ImagesLayerTest, TestResize) {
  LayerParameter param;
  param.set_batchsize(5);
  param.set_source(this->filename);
  param.set_new_height(256);
  param.set_new_width(256);
  param.set_shuffle_images(false);
  ImagesLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 256);
  EXPECT_EQ(this->blob_top_data_->width(), 256);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data 50 times
  for (int iter = 0; iter < 5; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(ImagesLayerTest, TestShuffle) {
  LayerParameter param;
  param.set_batchsize(5);
  param.set_source(this->filename);
  param.set_shuffle_images(true);
  ImagesLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 1200);
  EXPECT_EQ(this->blob_top_data_->width(), 1600);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data 5 times
  for (int iter = 0; iter < 5; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_GE(this->blob_top_label_->cpu_data()[i],0);
      EXPECT_LE(this->blob_top_label_->cpu_data()[i],5);
    }
  }
}
}
