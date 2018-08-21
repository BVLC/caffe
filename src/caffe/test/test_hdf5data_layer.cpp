#ifdef USE_HDF5
#include <string>
#include <vector>

#include "hdf5.h"

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/hdf5_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class HDF5DataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  HDF5DataLayerTest()
      : filename(NULL),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        blob_top_label2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    blob_top_vec_.push_back(blob_top_label2_);

    // Check out generate_sample_data.py in the same directory.
    filename = new string(
    CMAKE_SOURCE_DIR "caffe/test/test_data/sample_data_list.txt" CMAKE_EXT);
    LOG(INFO)<< "Using sample HDF5 data file " << filename;
  }

  virtual ~HDF5DataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
    delete blob_top_label2_;
    delete filename;
  }

  string* filename;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  Blob<Dtype>* const blob_top_label2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(HDF5DataLayerTest, TestDtypesAndDevices);

TYPED_TEST(HDF5DataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  // Create LayerParameter with the known parameters.
  // The data file we are reading has 10 rows and 8 columns,
  // with values from 0 to 10*8 reshaped in row-major order.
  LayerParameter param;
  param.add_top("data");
  param.add_top("label");
  param.add_top("label2");

  HDF5DataParameter* hdf5_data_param = param.mutable_hdf5_data_param();
  int batch_size = 5;
  hdf5_data_param->set_batch_size(batch_size);
  hdf5_data_param->set_source(*(this->filename));
  int num_cols = 8;
  int height = 6;
  int width = 5;

  // Test that the layer setup got the correct parameters.
  HDF5DataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), batch_size);
  EXPECT_EQ(this->blob_top_data_->channels(), num_cols);
  EXPECT_EQ(this->blob_top_data_->height(), height);
  EXPECT_EQ(this->blob_top_data_->width(), width);

  EXPECT_EQ(this->blob_top_label_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_label_->shape(0), batch_size);
  EXPECT_EQ(this->blob_top_label_->shape(1), 1);

  EXPECT_EQ(this->blob_top_label2_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_label2_->shape(0), batch_size);
  EXPECT_EQ(this->blob_top_label2_->shape(1), 1);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // Go through the data 10 times (5 batches).
  const int data_size = num_cols * height * width;
  for (int iter = 0; iter < 10; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // On even iterations, we're reading the first half of the data.
    // On odd iterations, we're reading the second half of the data.
    // NB: label is 1-indexed
    int label_offset = 1 + ((iter % 2 == 0) ? 0 : batch_size);
    int label2_offset = 1 + label_offset;
    int data_offset = (iter % 2 == 0) ? 0 : batch_size * data_size;

    // Every two iterations we are reading the second file,
    // which has the same labels, but data is offset by total data size,
    // which is 2400 (see generate_sample_data).
    int file_offset = (iter % 4 < 2) ? 0 : 2400;

    for (int i = 0; i < batch_size; ++i) {
      EXPECT_EQ(
        label_offset + i,
        this->blob_top_label_->cpu_data()[i]);
      EXPECT_EQ(
        label2_offset + i,
        this->blob_top_label2_->cpu_data()[i]);
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
              << " iter " << iter;
          }
        }
      }
    }
  }
}

}  // namespace caffe
#endif  // USE_HDF5
