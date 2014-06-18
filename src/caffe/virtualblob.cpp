// Copyright 2014 BVLC and contributors.

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"

namespace caffe {

template <typename Dtype>
VirtualBlob<Dtype>::VirtualBlob() {
  Reshape(0, 0, 0, 0);
}

template <typename Dtype>
VirtualBlob<Dtype>::VirtualBlob(const int num, const int channels, const int height,
    const int width) {
  Reshape(num, channels, height, width);
}

template <typename Dtype>
void VirtualBlob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
  CHECK_GE(num, 0);
  CHECK_GE(channels, 0);
  CHECK_GE(height, 0);
  CHECK_GE(width, 0);
  this->num_ = num;
  this->channels_ = channels;
  this->height_ = height;
  this->width_ = width;
  this->count_ = num * channels * height * width;
  if (this->count_ == 0) {
    // If new size is zero then release data and diff pointers
    this->data_.reset(reinterpret_cast<SyncedMemory*>(NULL));
    this->diff_.reset(reinterpret_cast<SyncedMemory*>(NULL));
  }
  if (this->data_) {
    // If it already has data then the new size has to fit in
    CHECK_GE(this->data_->size(), this->count_ * sizeof(Dtype));
  }
  if (this->diff_) {
    // If it already has diff then the new size has to fit in
    CHECK_GE(this->diff_->size(), this->count_ * sizeof(Dtype));
  }
}

template <typename Dtype>
void VirtualBlob<Dtype>::ShareData(const Blob<Dtype>& other) {
  // An VirtualBlob can share data with another Blobs that is bigger
  CHECK_LE(this->count_, other.count());
  this->data_ = other.data();
}

template <typename Dtype>
void VirtualBlob<Dtype>::ShareDiff(const Blob<Dtype>& other) {
  // An VirtualBlob can share diff with another Blobs that is bigger
  CHECK_LE(this->count_, other.count());
  this->diff_ = other.diff();
}

INSTANTIATE_CLASS(VirtualBlob);

}  // namespace caffe
