// Copyright 2013 Yangqing Jia

#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "leveldb/db.h"

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
class HDF5DataLayerTest : public ::testing::Test {
 protected:
  HDF5DataLayerTest()
      : blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        filename(NULL) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);

    // TODO: generate sample HDF5 file on the fly.
    // For now, use example HDF5 file.
    // TODO: how to best deal with the relativeness of the path?
    filename = "src/caffe/test/test_data/sample_data.h5";
    LOG(INFO) << "Using sample HDF5 data file " << filename;
  }

  virtual ~HDF5DataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  char* filename;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(HDF5DataLayerTest, Dtypes);

TYPED_TEST(HDF5DataLayerTest, TestRead) {
  // Create LayerParameter with the known parameters.
  // The data file we are reading has 10 rows and 8 columns,
  // with values from 0 to 10*8 reshaped in row-major order.
  LayerParameter param;
  int batchsize = 5;
  param.set_batchsize(batchsize);
  param.set_source(this->filename);
  int num_rows = 10;
  int num_cols = 8;
  HDF5DataLayer<TypeParam> layer(param);

  // Test that the layer setup got the correct parameters.
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), batchsize);
  EXPECT_EQ(this->blob_top_data_->channels(), num_cols);
  EXPECT_EQ(this->blob_top_data_->height(), 1);
  EXPECT_EQ(this->blob_top_data_->width(), 1);

  EXPECT_EQ(this->blob_top_label_->num(), batchsize);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  // Go through the data 100 times.
  for (int iter = 0; iter < 100; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);

    // On even iterations, we're reading the first half of the data.
    // On odd iterations, we're reading the second half of the data.
    int label_offset = (iter % 2 == 0) ? 0 : batchsize;
    int data_offset = (iter % 2 == 0) ? 0 : batchsize * num_cols;

    for (int i = 0; i < batchsize; ++i) {
      EXPECT_EQ(
        label_offset + i,
        this->blob_top_label_->cpu_data()[i]);
    }
    for (int i = 0; i < batchsize; ++i) {
      for (int j = 0; j < num_cols; ++j) {
        EXPECT_EQ(
          data_offset + i * num_cols + j,
          this->blob_top_data_->cpu_data()[i * num_cols + j])
          << "debug: i " << i << " j " << j;
      }
    }
  }

  // Exact same test in GPU mode.
  Caffe::set_mode(Caffe::GPU);
  // Go through the data 100 times.
  for (int iter = 0; iter < 100; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);

    // On even iterations, we're reading the first half of the data.
    // On odd iterations, we're reading the second half of the data.
    int label_offset = (iter % 2 == 0) ? 0 : batchsize;
    int data_offset = (iter % 2 == 0) ? 0 : batchsize * num_cols;

    for (int i = 0; i < batchsize; ++i) {
      EXPECT_EQ(
        label_offset + i,
        this->blob_top_label_->cpu_data()[i]);
    }
    for (int i = 0; i < batchsize; ++i) {
      for (int j = 0; j < num_cols; ++j) {
        EXPECT_EQ(
          data_offset + i * num_cols + j,
          this->blob_top_data_->cpu_data()[i * num_cols + j])
          << "debug: i " << i << " j " << j;
      }
    }
  }
}

}  // namespace caffe
