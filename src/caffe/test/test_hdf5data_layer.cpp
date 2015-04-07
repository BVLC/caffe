#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

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
    LOG(INFO) << "Using sample HDF5 data file " << *filename;
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

  void TestRead(int batch_size) {
    // Create LayerParameter with the known parameters.
    // The data file we are reading has 10 rows and 8 columns,
    // with values from 0 to 10*8 reshaped in row-major order.
    LayerParameter param;
    param.add_top("data");
    param.add_top("label");
    param.add_top("label2");
    // number of records in sample HDF5 data file (see generate_sample_data).
    const int num_data_records = 10;

    HDF5DataParameter* hdf5_data_param = param.mutable_hdf5_data_param();
    hdf5_data_param->set_batch_size(batch_size);
    hdf5_data_param->set_source(*(this->filename));
    const int num_cols = 8;
    const int height = 6;
    const int width = 5;

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

    // Go through the data 10 times.
    const int data_size = num_cols * height * width;
    for (int iter = 0; iter < 10; ++iter) {
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

      const int label_offset = iter * batch_size;
      const int data_offset = label_offset * data_size;

      for (int i = 0; i < batch_size; ++i) {
        // label is 1-indexed
        EXPECT_EQ(((label_offset + i) % num_data_records) + 1,
                  this->blob_top_label_->cpu_data()[i]);

        // label2 is 2-indexed
        EXPECT_EQ(((label_offset + i) % num_data_records) + 2,
                  this->blob_top_label2_->cpu_data()[i]);
      }
      for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_cols; ++j) {
          for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
              int idx = (i * num_cols * height * width +
                         j * height * width +
                         h * width + w);
              EXPECT_EQ((data_offset + idx)
                          % (2 * num_data_records * data_size),
                        this->blob_top_data_->cpu_data()[idx])
                << "debug: i " << i << " j " << j
                << " iter " << iter;
            }
          }
        }
      }
    }
  }
};

TYPED_TEST_CASE(HDF5DataLayerTest, TestDtypesAndDevices);

TYPED_TEST(HDF5DataLayerTest, TestRead) {
  this->TestRead(5);
}

TYPED_TEST(HDF5DataLayerTest, TestInterleavingRead) {
  this->TestRead(3);
}


}  // namespace caffe
