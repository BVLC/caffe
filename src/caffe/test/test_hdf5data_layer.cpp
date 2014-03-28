// Copyright 2014 BVLC and contributors.

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

    // Check out generate_sample_data.py in the same directory.
    filename = new string("src/caffe/test/test_data/sample_data_list.txt");
    LOG(INFO) << "Using sample HDF5 data file " << filename;
  }

  virtual ~HDF5DataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
    delete filename;
  }

  string* filename;
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
  HDF5DataParameter* hdf5_data_param = param.mutable_hdf5_data_param();
  int batch_size = 5;
  hdf5_data_param->set_batch_size(batch_size);
  hdf5_data_param->set_source(*(this->filename));
  int num_rows = 10;
  int num_cols = 8;
  int height = 5;
  int width = 5;

  // Test that the layer setup got the correct parameters.
  HDF5DataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), batch_size);
  EXPECT_EQ(this->blob_top_data_->channels(), num_cols);
  EXPECT_EQ(this->blob_top_data_->height(), height);
  EXPECT_EQ(this->blob_top_data_->width(), width);

  EXPECT_EQ(this->blob_top_label_->num(), batch_size);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  for (int t = 0; t < 2; ++t) {
    // TODO: make this a TypedTest instead of this silly loop.
    if (t == 0) {
      Caffe::set_mode(Caffe::CPU);
    } else {
      Caffe::set_mode(Caffe::GPU);
    }
    layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);

    // Go through the data 10 times (5 batches).
    const int data_size = num_cols * height * width;
    for (int iter = 0; iter < 10; ++iter) {
      layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);

      // On even iterations, we're reading the first half of the data.
      // On odd iterations, we're reading the second half of the data.
      int label_offset = (iter % 2 == 0) ? 0 : batch_size;
      int data_offset = (iter % 2 == 0) ? 0 : batch_size * data_size;

      // Every two iterations we are reading the second file,
      // which has the same labels, but data is offset by total data size,
      // which is 2000 (see generate_sample_data).
      int file_offset = (iter % 4 < 2) ? 0 : 2000;

      for (int i = 0; i < batch_size; ++i) {
        EXPECT_EQ(
          label_offset + i,
          this->blob_top_label_->cpu_data()[i]);
      }
      for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_cols; ++j) {
          for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
              int idx = (
                i * num_cols * height * width +
                j * height * width +
                h * width + w);
              EXPECT_EQ(
                file_offset + data_offset + idx,
                this->blob_top_data_->cpu_data()[idx])
                << "debug: i " << i << " j " << j
                << " iter " << iter << " t " << t;
            }
          }
        }
      }
    }
  }
}

}  // namespace caffe
