// Copyright 2014 BVLC and contributors.

#include <cstring>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/objdetect/rect.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class BlobTest : public ::testing::Test {
 protected:
  BlobTest()
      : blob_(new Blob<Dtype>()),
        blob_preshaped_(new Blob<Dtype>(2, 3, 4, 5)) {}

  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_preshaped_);
  }

  virtual ~BlobTest() { delete blob_; delete blob_preshaped_; }
  Blob<Dtype>* const blob_;
  Blob<Dtype>* const blob_preshaped_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(BlobTest, Dtypes);

TYPED_TEST(BlobTest, TestInitialization) {
  EXPECT_TRUE(this->blob_);
  EXPECT_TRUE(this->blob_preshaped_);
  EXPECT_EQ(this->blob_preshaped_->num(), 2);
  EXPECT_EQ(this->blob_preshaped_->channels(), 3);
  EXPECT_EQ(this->blob_preshaped_->height(), 4);
  EXPECT_EQ(this->blob_preshaped_->width(), 5);
  EXPECT_EQ(this->blob_preshaped_->count(), 120);
  EXPECT_EQ(this->blob_->num(), 0);
  EXPECT_EQ(this->blob_->channels(), 0);
  EXPECT_EQ(this->blob_->height(), 0);
  EXPECT_EQ(this->blob_->width(), 0);
  EXPECT_EQ(this->blob_->count(), 0);
}

TYPED_TEST(BlobTest, TestPointersCPU) {
  EXPECT_TRUE(this->blob_preshaped_->cpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_cpu_data());
}

TYPED_TEST(BlobTest, TestPointersGPU) {
  EXPECT_TRUE(this->blob_preshaped_->gpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_gpu_data());
}

TYPED_TEST(BlobTest, TestReshape) {
  this->blob_->Reshape(2, 3, 4, 5);
  EXPECT_EQ(this->blob_->num(), 2);
  EXPECT_EQ(this->blob_->channels(), 3);
  EXPECT_EQ(this->blob_->height(), 4);
  EXPECT_EQ(this->blob_->width(), 5);
  EXPECT_EQ(this->blob_->count(), 120);
}

TYPED_TEST(BlobTest, TestCopyFromRegionCPU) {
  Caffe::set_mode(Caffe::CPU);
  Rect region(1, 1, 3, 4);
  this->blob_->CopyFromRegion(*(this->blob_preshaped_), region, false, true);
  EXPECT_EQ(this->blob_->num(), 2);
  EXPECT_EQ(this->blob_->channels(), 3);
  EXPECT_EQ(this->blob_->height(), 3);
  EXPECT_EQ(this->blob_->width(), 2);
  EXPECT_EQ(this->blob_->count(), 36);
  for (int n = 0; n < 2; ++n) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < 3; ++h) {
        for (int w = 0; w < 2; ++w) {
          EXPECT_EQ(this->blob_preshaped_->data_at(n, c, h + 1, w + 1),
                    this->blob_->data_at(n, c, h, w));
        }
      }
    }
  }
  Rect region2(1, 2, 4, 5);
  caffe_copy(this->blob_preshaped_->count(), this->blob_preshaped_->cpu_data(),
             this->blob_preshaped_->mutable_cpu_diff());
  this->blob_->CopyFromRegion(*(this->blob_preshaped_), region2, true, true);
  EXPECT_EQ(this->blob_->num(), 2);
  EXPECT_EQ(this->blob_->channels(), 3);
  EXPECT_EQ(this->blob_->height(), 3);
  EXPECT_EQ(this->blob_->width(), 3);
  EXPECT_EQ(this->blob_->count(), 54);
  for (int n = 0; n < 2; ++n) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < 3; ++h) {
        for (int w = 0; w < 3; ++w) {
          EXPECT_EQ(this->blob_preshaped_->diff_at(n, c, h + 2, w + 1),
                    this->blob_->data_at(n, c, h, w));
        }
      }
    }
  }
}

}  // namespace caffe
