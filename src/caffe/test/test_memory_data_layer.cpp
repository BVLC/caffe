// Copyright 2014 kloudkl@github

#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template<typename Dtype>
class MemoryDataLayerTest : public ::testing::Test {
 protected:
  MemoryDataLayerTest()
      : batch_size_(64),
        channels_(11),
        height_(17),
        width_(19),
        num_labels_(10),
        blob_bottom_data_(
            new Blob<Dtype>(batch_size_, channels_, height_, width_)),
        blob_bottom_label_(new Blob<Dtype>(batch_size_, num_labels_, 1, 1)),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {
  }

  virtual void SetUp() {
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
  }

  virtual ~MemoryDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  int batch_size_;
  int channels_;
  int height_;
  int width_;
  int num_labels_;
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(MemoryDataLayerTest, Dtypes);

TYPED_TEST(MemoryDataLayerTest, TestSetup){
  LayerParameter param;
  MemoryDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), this->batch_size_);
  EXPECT_EQ(this->blob_top_data_->channels(), this->channels_);
  EXPECT_EQ(this->blob_top_data_->height(), this->height_);
  EXPECT_EQ(this->blob_top_data_->width(), this->width_);
  EXPECT_EQ(this->blob_top_label_->num(), this->batch_size_);
  EXPECT_EQ(this->blob_top_label_->channels(), this->num_labels_);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
}

TYPED_TEST(MemoryDataLayerTest, TestForward){
  LayerParameter param;
  MemoryDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);

  FillerParameter filler_param;
  GaussianFiller<TypeParam> filler(filler_param);
  Caffe::Brew modes[] = {Caffe::CPU, Caffe::GPU};
  for (int n_mode = 0; n_mode < 2; ++n_mode) {
    Caffe::set_mode(modes[n_mode]);
    for (int iter = 0; iter < 100; ++iter) {
      filler.Fill(this->blob_bottom_data_);
      filler.Fill(this->blob_bottom_label_);
      layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
      for (int i = 0; i < this->blob_bottom_data_->count(); ++i) {
        EXPECT_EQ(this->blob_bottom_data_->cpu_data()[i],
                  this->blob_bottom_data_->cpu_data()[i]);
      }
      for (int i = 0; i < this->blob_bottom_label_->count(); ++i) {
        EXPECT_EQ(this->blob_bottom_label_->cpu_data()[i],
                  this->blob_top_label_->cpu_data()[i]);
      }
    }
  }
}

TYPED_TEST(MemoryDataLayerTest, TestDynamicBatchSize){
  LayerParameter param;
  MemoryDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);

  FillerParameter filler_param;
  GaussianFiller<TypeParam> filler(filler_param);
  Caffe::Brew modes[] = {Caffe::CPU, Caffe::GPU};
  for (int n_mode = 0; n_mode < 2; ++n_mode) {
    Caffe::set_mode(modes[n_mode]);
    for (int batch_size = 1; batch_size <= this->batch_size_; ++batch_size) {
      this->blob_bottom_data_->Reshape(batch_size, this->channels_,
                                       this->height_, this->width_);
      filler.Fill(this->blob_bottom_data_);
      filler.Fill(this->blob_bottom_label_);
      layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
      for (int i = 0; i < this->blob_bottom_data_->count(); ++i) {
        EXPECT_EQ(this->blob_bottom_data_->cpu_data()[i],
                  this->blob_bottom_data_->cpu_data()[i]);
      }
      for (int i = 0; i < this->blob_bottom_label_->count(); ++i) {
        EXPECT_EQ(this->blob_bottom_label_->cpu_data()[i],
                  this->blob_top_label_->cpu_data()[i]);
      }
    }
  }
}

}
  // namespace caffe
