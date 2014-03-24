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
using std::stringstream;

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class DataLayerTest : public ::testing::Test {
 protected:
  DataLayerTest()
      : blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        filename(NULL) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    // Create the leveldb
    filename = tmpnam(NULL);  // get temp name
    LOG(INFO) << "Using temporary leveldb " << filename;
    leveldb::DB* db;
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    leveldb::Status status = leveldb::DB::Open(options, filename, &db);
    CHECK(status.ok());
    for (int i = 0; i < 5; ++i) {
      Datum datum;
      datum.set_label(i);
      datum.set_channels(2);
      datum.set_height(3);
      datum.set_width(4);
      std::string* data = datum.mutable_data();
      for (int j = 0; j < 24; ++j) {
        data->push_back((uint8_t)i);
      }
      stringstream ss;
      ss << i;
      db->Put(leveldb::WriteOptions(), ss.str(), datum.SerializeAsString());
    }
    delete db;
  }

  virtual ~DataLayerTest() { delete blob_top_data_; delete blob_top_label_; }

  char* filename;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(DataLayerTest, Dtypes);

TYPED_TEST(DataLayerTest, TestRead) {
  LayerParameter param;
  param.set_batchsize(5);
  param.set_source(this->filename);
  DataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 2);
  EXPECT_EQ(this->blob_top_data_->height(), 3);
  EXPECT_EQ(this->blob_top_data_->width(), 4);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  // Go through the data 100 times
  for (int iter = 0; iter < 100; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 24; ++j) {
        EXPECT_EQ(i, this->blob_top_data_->cpu_data()[i * 24 + j])
            << "debug: i " << i << " j " << j;
      }
    }
  }

  // Same test, in GPU mode.
  Caffe::set_mode(Caffe::GPU);
  for (int iter = 0; iter < 100; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 24; ++j) {
        EXPECT_EQ(i, this->blob_top_data_->cpu_data()[i * 24 + j])
            << "debug: i " << i << " j " << j;
      }
    }
  }
}

}  // namespace caffe
