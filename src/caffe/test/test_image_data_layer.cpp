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
      : seed_(1701),
        filename_(new string(tmpnam(NULL))),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
  }

  void FillImageList(const int num_labels = 1) {
    std::ofstream outfile(filename_->c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << *filename_;
    switch (num_labels) {
    case 0:
      // Images without labels
      for (int i = 0; i < 5; ++i) {
        outfile << "examples/images/cat.jpg " << endl;
      }
      break;
    case 1:
      // Create a List of files with a single label
      for (int i = 0; i < 5; ++i) {
        outfile << "examples/images/cat.jpg " << i << endl;
      }
      break;
    default:
      // Create a Vector of files with muliple {-1,1} labels
      for (int i = 0; i < 5; ++i) {
        outfile << "examples/images/cat.jpg ";
        for (int l = 0; l < num_labels; ++l) {
          if (l == i) {
            outfile << " 1";
          } else {
            outfile << " -1";
          }
        }
        outfile << endl;
      }
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
  const int num_labels = 1;
  this->FillImageList(num_labels);
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_->c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), num_labels);
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

TYPED_TEST(ImageDataLayerTest, TestReadWithoutLabels) {
  const int num_labels = 0;
  this->FillImageList(num_labels);
  vector<Blob<TypeParam>*> aux_blob_top_vec_;
  aux_blob_top_vec_.push_back(this->blob_top_data_);
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_->c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &aux_blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(aux_blob_top_vec_.size(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &aux_blob_top_vec_);
    EXPECT_EQ(aux_blob_top_vec_.size(), 1);
  }
}

TYPED_TEST(ImageDataLayerTest, TestReadMultiLabel) {
  const int num_labels = 5;
  this->FillImageList(num_labels);
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_->c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), num_labels);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      for (int l = 0; l < 5; ++l) {
        if (i == l) {
          EXPECT_EQ(1, this->blob_top_label_->cpu_data()[i * 5 + l]);
        } else {
          EXPECT_EQ(-1, this->blob_top_label_->cpu_data()[i * 5 + l ]);
        }
      }
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestResize) {
  const int num_labels = 1;
  this->FillImageList(num_labels);
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
  EXPECT_EQ(this->blob_top_label_->channels(), num_labels);
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
  Caffe::set_random_seed(this->seed_);
  const int num_labels = 1;
  this->FillImageList(num_labels);
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_->c_str());
  image_data_param->set_shuffle(true);
  ImageDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), num_labels);
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
