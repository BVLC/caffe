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
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() {
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return (Dtype*)data_->mutable_cpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return (Dtype*)data_->mutable_gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return (Dtype*)diff_->mutable_cpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return (Dtype*)diff_->mutable_gpu_data();
}

template <typename Dtype>
void Blob<Dtype>::update() {
  
}

template class Blob<float>;
template class Blob<double>;

}  // namespace caffeine

