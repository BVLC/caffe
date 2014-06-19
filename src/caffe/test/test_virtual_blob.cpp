// Copyright 2014 BVLC and contributors.

#include <cstring>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class VirtualBlobSimpleTest : public ::testing::Test {
 protected:
  VirtualBlobSimpleTest()
      : virtual_blob_(new VirtualBlob<Dtype>()),
        real_blob_(new Blob<Dtype>(2, 3, 4, 5)) {}
  virtual ~VirtualBlobSimpleTest() { delete virtual_blob_; delete real_blob_; }
  VirtualBlob<Dtype>* const virtual_blob_;
  Blob<Dtype>* const real_blob_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(VirtualBlobSimpleTest, Dtypes);

TYPED_TEST(VirtualBlobSimpleTest, TestInitialization) {
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

TYPED_TEST(VirtualBlobSimpleTest, TestPointers) {
  EXPECT_TRUE(this->real_blob_->gpu_data());
  EXPECT_TRUE(this->real_blob_->cpu_data());
  EXPECT_TRUE(this->real_blob_->mutable_gpu_data());
  EXPECT_TRUE(this->real_blob_->mutable_cpu_data());
}

TYPED_TEST(VirtualBlobSimpleTest, TestReshape) {
  this->virtual_blob_->Reshape(2, 3, 4, 5);
  EXPECT_EQ(this->virtual_blob_->num(), 2);
  EXPECT_EQ(this->virtual_blob_->channels(), 3);
  EXPECT_EQ(this->virtual_blob_->height(), 4);
  EXPECT_EQ(this->virtual_blob_->width(), 5);
  EXPECT_EQ(this->virtual_blob_->count(), 120);
}

TYPED_TEST(VirtualBlobSimpleTest, TestShareData) {
  this->virtual_blob_->Reshape(2, 3, 4, 5);
  this->virtual_blob_->ShareData(*(this->real_blob_));
  EXPECT_EQ(this->virtual_blob_->data(), this->real_blob_->data());
}

TYPED_TEST(VirtualBlobSimpleTest, TestShareDataSize) {
  this->virtual_blob_->Reshape(1, 1, 1, 1);
  this->virtual_blob_->ShareData(*(this->real_blob_));
  EXPECT_EQ(this->virtual_blob_->data()->size(),
    this->real_blob_->data()->size());
}

TYPED_TEST(VirtualBlobSimpleTest, TestShareDif) {
  this->virtual_blob_->Reshape(2, 3, 4, 5);
  this->virtual_blob_->ShareDiff(*(this->real_blob_));
  EXPECT_EQ(this->virtual_blob_->diff(), this->real_blob_->diff());
}

TYPED_TEST(VirtualBlobSimpleTest, TestShareDifSize) {
  this->virtual_blob_->Reshape(1, 1, 1, 1);
  this->virtual_blob_->ShareDiff(*(this->real_blob_));
  EXPECT_EQ(this->virtual_blob_->diff()->size(),
    this->real_blob_->diff()->size());
}

}  // namespace caffe
