// Copyright 2014 BVLC and contributors.

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"

namespace caffe {

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
  num_ = num;
  channels_ = channels;
  height_ = height;
  width_ = width;
  count_ = num_ * channels_ * height_ * width_;
  if (count_ == 0) {
    // If new size is zero then release data and diff pointers
    data_.reset(reinterpret_cast<SyncedMemory*>(NULL));
    diff_.reset(reinterpret_cast<SyncedMemory*>(NULL));
  }
  if (data_) {
    // If it already has data then the new size has to fit in
    CHECK_GE(data_.size(), count_ * sizeof(Dtype));
  }
  if (diff_) {
    // If it already has diff then the new size has to fit in
    CHECK_GE(diff_.size(), count_ * sizeof(Dtype));
  }
}

template <typename Dtype>
void VirtualBlob<Dtype>::ShareData(const Blob<Dtype>& other) {
  // An VirtualBlob can share data with another Blobs that is bigger
  CHECK_LE(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void VirtualBlob<Dtype>::ShareDiff(const Blob<Dtype>& other) {
  // An VirtualBlob can share diff with another Blobs that is bigger
  CHECK_LE(count_, other.count());
  diff_ = other.diff();
}

INSTANTIATE_CLASS(VirtualBlob);

}  // namespace caffe
