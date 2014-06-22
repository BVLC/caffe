// Copyright 2014 BVLC and contributors.

#include <cuda_runtime.h>

#include <iostream>  // NOLINT(readability/streams)
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/test/test_caffe_main.hpp"

using std::map;
using std::string;

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class ImageDataLayerTest : public ::testing::Test {
 protected:
  ImageDataLayerTest()
      : blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        filename_(new string(tmpnam(NULL))),
        seed_(1701) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);
    // Create a Vector of files with labels
    std::ofstream outfile(filename_->c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << *filename_;
    for (int i = 0; i < 5; ++i) {
      outfile << "examples/images/cat.jpg " << i;
    }
    outfile.close();
  }

  virtual ~ImageDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  int seed_;
  shared_ptr<string> filename_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(ImageDataLayerTest, Dtypes);

TYPED_TEST(ImageDataLayerTest, TestRead) {
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_->c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 1200);
  EXPECT_EQ(this->blob_top_data_->width(), 1600);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestResize) {
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_->c_str());
  image_data_param->set_new_height(256);
  image_data_param->set_new_width(256);
  image_data_param->set_shuffle(false);
  ImageDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 256);
  EXPECT_EQ(this->blob_top_data_->width(), 256);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestShuffle) {
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_->c_str());
  image_data_param->set_shuffle(true);
  ImageDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 1200);
  EXPECT_EQ(this->blob_top_data_->width(), 1600);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    map<TypeParam, int> values_to_indices;
    int num_in_order = 0;
    for (int i = 0; i < 5; ++i) {
      TypeParam value = this->blob_top_label_->cpu_data()[i];
      // Check that the value has not been seen already (no duplicates).
      EXPECT_EQ(values_to_indices.find(value), values_to_indices.end());
      values_to_indices[value] = i;
      num_in_order += (value == TypeParam(i));
    }
    EXPECT_EQ(5, values_to_indices.size());
    EXPECT_GT(5, num_in_order);
  }
}

}  // namespace caffe
