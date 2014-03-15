// Copyright 2013 Yangqing Jia

#include <cuda_runtime.h>
#include <cstring>

#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class BlobSimpleTest : public ::testing::Test {
 protected:
  BlobSimpleTest()
      : blob_(new Blob<Dtype>()),
        blob_preshaped_(new Blob<Dtype>(2, 3, 4, 5)) {}
  virtual ~BlobSimpleTest() { delete blob_; delete blob_preshaped_; }
  Blob<Dtype>* const blob_;
  Blob<Dtype>* const blob_preshaped_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(BlobSimpleTest, Dtypes);

TYPED_TEST(BlobSimpleTest, TestInitialization) {
  EXPECT_TRUE(this->blob_);
  EXPECT_TRUE(this->blob_preshaped_);
  EXPECT_EQ(this->blob_preshaped_->num(), 2);
  EXPECT_EQ(this->blob_preshaped_->channels(), 3);
  EXPECT_EQ(this->blob_preshaped_->height(), 4);
  EXPECT_EQ(this->blob_preshaped_->width(), 5);
  EXPECT_EQ(this->blob_preshaped_->count(), 120);
  EXPECT_EQ(this->blob_preshaped_->data_size(), 120 * sizeof(TypeParam));
  EXPECT_EQ(this->blob_preshaped_->diff_size(), 120 * sizeof(TypeParam));
  EXPECT_EQ(this->blob_->num(), 0);
  EXPECT_EQ(this->blob_->channels(), 0);
  EXPECT_EQ(this->blob_->height(), 0);
  EXPECT_EQ(this->blob_->width(), 0);
  EXPECT_EQ(this->blob_->count(), 0);
  EXPECT_FALSE(this->blob_->has_data());
  EXPECT_FALSE(this->blob_->has_diff());
}

TYPED_TEST(BlobSimpleTest, TestPointers) {
  EXPECT_TRUE(this->blob_preshaped_->gpu_data());
  EXPECT_TRUE(this->blob_preshaped_->cpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_gpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_cpu_data());
}

TYPED_TEST(BlobSimpleTest, TestReshape) {
  this->blob_->Reshape(1, 2, 3, 4);
  EXPECT_EQ(this->blob_->num(), 1);
  EXPECT_EQ(this->blob_->channels(), 2);
  EXPECT_EQ(this->blob_->height(), 3);
  EXPECT_EQ(this->blob_->width(), 4);
  EXPECT_EQ(this->blob_->count(), 24);
  EXPECT_EQ(this->blob_->data_size(), 24 * sizeof(TypeParam));
  EXPECT_EQ(this->blob_->diff_size(), 24 * sizeof(TypeParam));
}

TYPED_TEST(BlobSimpleTest, TestReserve) {
  const int capacity = 100;
  const int size_of_data_type = sizeof(TypeParam);
  this->blob_->Reserve(capacity);
  EXPECT_EQ(this->blob_->capacity(), capacity);
  EXPECT_EQ(this->blob_->data_size(), capacity * size_of_data_type);
  EXPECT_EQ(this->blob_->diff_size(), capacity * size_of_data_type);

  this->blob_->Reserve(capacity - 1);
  EXPECT_EQ(this->blob_->capacity(), capacity);
  EXPECT_EQ(this->blob_->data_size(), capacity * size_of_data_type);
  EXPECT_EQ(this->blob_->diff_size(), capacity * size_of_data_type);

  this->blob_->Reserve(capacity + 1);
  EXPECT_EQ(this->blob_->capacity(), capacity + 1);
  EXPECT_EQ(this->blob_->data_size(), (capacity + 1) * size_of_data_type);
  EXPECT_EQ(this->blob_->diff_size(), (capacity + 1) * size_of_data_type);

  this->blob_->Reserve(0);
  EXPECT_EQ(this->blob_->capacity(), 0);
  EXPECT_FALSE(this->blob_->has_data());
  EXPECT_FALSE(this->blob_->has_diff());
}

}  // namespace caffe
