// Copyright 2014 kloudkl@github

#include <H5Cpp.h>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"

namespace caffe {
using namespace H5;

template <typename Dtype>
class IOTest : public ::testing::Test {
 protected:
  IOTest(): num_(11), channels_(17), height_(19),
  width_(23), hdf5_file_name_("src/caffe/test/test_data/write_blob.h5"),
  hdf5_dataset_name_("hdn"), data_type_(PredType::NATIVE_FLOAT),
  blob_(num_, channels_, height_, width_) {}

  virtual void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_);
    if (sizeof(Dtype) == sizeof(float)) {
      data_type_ = PredType::NATIVE_FLOAT;
    } else {
      data_type_ = PredType::NATIVE_DOUBLE;
    }
  }

  virtual ~IOTest() {}

  int num_;
  int channels_;
  int height_ ;
  int width_;
  string hdf5_file_name_;
  string hdf5_dataset_name_;
  PredType data_type_;
  Blob<Dtype> blob_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(IOTest, Dtypes);

TYPED_TEST(IOTest, TestWriteAndReadBlobToHDF5File) {
  WriteBlobToHDF5File<TypeParam>(this->blob_, this->hdf5_file_name_,
                      this->hdf5_dataset_name_, this->data_type_);
  Blob<TypeParam> blob;
  ReadBlobFromHDF5File<TypeParam>(
      this->hdf5_file_name_, this->hdf5_dataset_name_, &blob);
  EXPECT_EQ(blob.num(), this->num_);
  EXPECT_EQ(blob.channels(), this->channels_);
  EXPECT_EQ(blob.height(), this->height_);
  EXPECT_EQ(blob.width(), this->width_);
  const TypeParam* data_gt = this->blob_.cpu_data();
  const TypeParam* data = blob.cpu_data();
  int idx = 0;
  for (int i = 0; i < this->num_; ++i) {
    for (int j = 0; j < this->channels_; ++j) {
      for (int h = 0; h < this->height_; ++h) {
        for (int w = 0; w < this->width_; ++w, ++idx) {
          EXPECT_EQ(data[idx], data_gt[idx])
              << "debug: i " << i << " j " << j << " h " << h << " w " << w;
        }
      }
    }
  }
}

}  // namespace caffe
