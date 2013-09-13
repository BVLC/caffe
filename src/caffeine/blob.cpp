#include "caffeine/blob.hpp"
#include "caffeine/common.hpp"
#include "caffeine/syncedmem.hpp"

namespace caffeine {

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
  num_ = num;
  channels_ = channels;
  height_ = height;
  width_ = width;
  count_ = num_ * channels_ * height_ * width_;
  data_.reset(new SyncedMemory(count_ * sizeof(Dtype)));
  diff_.reset(new SyncedMemory(count_ * sizeof(Dtype)));
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() {
  check_data();
  return data_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() {
  check_data();
  return data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() {
  check_diff();
  return diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() {
  check_diff();
  return diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  check_data();
  return data_->mutable_cpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  check_data();
  return data_->mutable_gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  check_diff();
  return diff_->mutable_cpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  check_diff();
  return diff_->mutable_gpu_data();
}

template <typename Dtype>
void Blob<Dtype>::update() {
  
}

}  // namespace caffeine
