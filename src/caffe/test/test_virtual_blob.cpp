// Copyright 2014 BVLC and contributors.

#include <cstring>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class BlobSimpleTest : public ::testing::Test {
 protected:
  BlobSimpleTest()
      : virtual_blob_(new VirtualBlob<Dtype>()),
        real_blob_(new Blob<Dtype>(2, 3, 4, 5)) {}
  virtual ~BlobSimpleTest() { delete virtual_blob_; delete real_blob_; }
  Blob<Dtype>* const virtual_blob_;
  Blob<Dtype>* const real_blob_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(BlobSimpleTest, Dtypes);

TYPED_TEST(BlobSimpleTest, TestInitialization) {
  EXPECT_TRUE(this->virtual_blob_);
  EXPECT_TRUE(this->real_blob_);
  EXPECT_EQ(this->real_blob_->num(), 2);
  EXPECT_EQ(this->real_blob_->channels(), 3);
  EXPECT_EQ(this->real_blob_->height(), 4);
  EXPECT_EQ(this->real_blob_->width(), 5);
  EXPECT_EQ(this->real_blob_->count(), 120);
  EXPECT_EQ(this->virtual_blob_->num(), 0);
  EXPECT_EQ(this->virtual_blob_->channels(), 0);
  EXPECT_EQ(this->virtual_blob_->height(), 0);
  EXPECT_EQ(this->virtual_blob_->width(), 0);
  EXPECT_EQ(this->virtual_blob_->count(), 0);
}

TYPED_TEST(BlobSimpleTest, TestPointers) {
  EXPECT_TRUE(this->real_blob_->gpu_data());
  EXPECT_TRUE(this->real_blob_->cpu_data());
  EXPECT_TRUE(this->real_blob_->mutable_gpu_data());
  EXPECT_TRUE(this->real_blob_->mutable_cpu_data());
}

TYPED_TEST(BlobSimpleTest, TestReshape) {
  this->virtual_blob_->Reshape(2, 3, 4, 5);
  EXPECT_EQ(this->virtual_blob_->num(), 2);
  EXPECT_EQ(this->virtual_blob_->channels(), 3);
  EXPECT_EQ(this->virtual_blob_->height(), 4);
  EXPECT_EQ(this->virtual_blob_->width(), 5);
  EXPECT_EQ(this->virtual_blob_->count(), 120);
}

TYPED_TEST(BlobSimpleTest, TestShareData) {
  this->virtual_blob_->Reshape(2, 3, 4, 5);
  this->virtual_blob_->ShareData(this->real_blob_)
  EXPECT_EQ(this->virtual_blob_->gpu_data(),this->real_blob_->gpu_data());
  EXPECT_EQ(this->virtual_blob_->cpu_data(),this->real_blob_->cpu_data());
  EXPECT_EQ(this->virtual_blob_->mutable_gpu_data(),this->real_blob_->mutable_gpu_data());
  EXPECT_EQ(this->virtual_blob_->mutable_cpu_data(),this->real_blob_->mutable_cpu_data());
}

TYPED_TEST(BlobSimpleTest, TestShareDif) {
  this->virtual_blob_->Reshape(2, 3, 4, 5);
  this->virtual_blob_->ShareDiff(this->real_blob_)
  EXPECT_EQ(this->virtual_blob_->gpu_diff(),this->real_blob_->gpu_diff());
  EXPECT_EQ(this->virtual_blob_->cpu_diff(),this->real_blob_->cpu_diff());
  EXPECT_EQ(this->virtual_blob_->mutable_gpu_diff(),this->real_blob_->mutable_gpu_diff());
  EXPECT_EQ(this->virtual_blob_->mutable_cpu_diff(),this->real_blob_->mutable_cpu_diff());
}

}  // namespace caffe
