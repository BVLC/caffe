#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"


namespace caffe {

  template <class T>
  string givelistfilename();

  template <>
  string givelistfilename<float>() {
    string dir(CMAKE_SOURCE_DIR "caffe/test/test_data/");
    string basename("categorical_hdf5_sample_data_list_float.txt"  CMAKE_EXT);
    string filename = dir+basename;
    return filename;
  }
  template <>
  string givelistfilename<double>() {
    string dir(CMAKE_SOURCE_DIR "caffe/test/test_data/");
    string basename("categorical_hdf5_sample_data_list_double.txt" CMAKE_EXT);
    string filename = dir+basename;
    return filename;
  }


  template <typename TypeParam>
  class HDF5CategoricalDLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;

   protected:
    HDF5CategoricalDLayerTest() : filename(NULL),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
    virtual void SetUp() {
      blob_top_vec_.push_back(blob_top_data_);
      blob_top_vec_.push_back(blob_top_label_);
      filename = new string(givelistfilename<Dtype>());
      // this numbers were used to generate test data
      numcategories = 5;
      numsamples = 50;
      // description vector is similar to  HDF5CategoricalDLayer.description
      // but slightly different
      description.resize(numcategories);
      for (int i = 0; i < numcategories; ++i)
        description[i]=i+1;
    }

    virtual ~HDF5CategoricalDLayerTest() {
      delete blob_top_data_;
      delete blob_top_label_;
      delete filename;
    }

    string* filename;
    Blob<Dtype>* const blob_top_data_;
    Blob<Dtype>* const blob_top_label_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
    vector<int> description;
    int numcategories;
    int numsamples;
  };

TYPED_TEST_CASE(HDF5CategoricalDLayerTest, TestDtypesAndDevices);

TYPED_TEST(HDF5CategoricalDLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  // Create LayerParameter with the known parameters.
  LayerParameter param;
  param.add_top("data");
  param.add_top("label");

  HDF5CategoricalDParameter* hdf5_categorical_data_param=
    param.mutable_hdf5_categorical_data_param();
  int batch_size = 20;
  hdf5_categorical_data_param->set_batch_size(batch_size);
  hdf5_categorical_data_param->set_source(*(this->filename));
  // see 'generate_test_data'
  int numcategories = this->numcategories;
  int num_cols = (numcategories+1)*numcategories/2;
  int height = 1;
  int width = 1;

  // Test that the layer setup got the correct parameters.
  HDF5CategoricalDLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), batch_size);
  EXPECT_EQ(this->blob_top_data_->channels(), num_cols);
  EXPECT_EQ(this->blob_top_data_->height(), height);
  EXPECT_EQ(this->blob_top_data_->width(), width);

  EXPECT_EQ(this->blob_top_label_->num(), batch_size);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  // helper data to check one-hot-encoding
  vector<int> accummulatedvalues(numcategories);
  accummulatedvalues[0] = 0;
  for (int i = 1; i <numcategories; ++i)
    accummulatedvalues[i] = accummulatedvalues[i-1]+i;

  int numsamples = this->numsamples;
  int datasample, value, idx;
  int numiterations = 6;
  for (int iter = 0; iter < numiterations; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    for (int i = 0; i < batch_size; ++i) {
      datasample = (iter*batch_size+i)%numsamples;
      // verify one-hot-encoded batch data  (see  'generate_test_data')
      for (int j = 0; j < numcategories; ++j) {
        // first data entry corresponding to this category in this row
        idx = i*num_cols + accummulatedvalues[j];
        // check if this category is 'set' for this datasample
        if (datasample%(j+2)) {
          value = datasample%(j+2)-1;
          // there are j possible values for category j
          for (int k = 0; k < j; ++k)
            if (k == value)
              ASSERT_EQ(1, this->blob_top_data_->cpu_data()[idx+k]);
            else
              ASSERT_EQ(0, this->blob_top_data_->cpu_data()[idx+k]);
         } else {
          // no value is set for this category
          for (int k = 0; k < j; ++k)
            ASSERT_EQ(0, this->blob_top_data_->cpu_data()[idx+k]);
        }
      }

      // check label value
      ASSERT_EQ(static_cast<float>(datasample%2),
                this->blob_top_label_->cpu_data()[i]);
    }
  }
}

}  // namespace caffe
