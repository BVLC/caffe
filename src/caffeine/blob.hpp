#ifndef CAFFEINE_BLOB_HPP
#define CAFFEINE_BLOB_HPP

#include <memory>

#include "caffeine/common.hpp"
#include "caffeine/syncedmem.hpp"

namespace caffeine {

template <typename Dtype>
class Blob {
 public:
  Blob()
       : num_(0), channels_(0), height_(0), width_(0), count_(0), data_(),
       diff_() {};
  explicit Blob(const int num, const int channels, const int height,
    const int width)
      : num_(num), channels_(channels), height_(height), width_(width),
      count_(num * channels * height * width),
      data_(new SyncedMemory(count_ * sizeof(Dtype))),
      diff_(new SyncedMemory(count_ * sizeof(Dtype))) {};
  virtual ~Blob() {};
  void Reshape(const int num, const int channels, const int height,
      const int width);
  inline int num() { return num_; }
  inline int channels() { return channels_; }
  inline int height() { return height_; }
  inline int width() { return width_; }
  inline int count() {return count_; }
  
  const Dtype* cpu_data();
  const Dtype* gpu_data();
  const Dtype* cpu_diff();
  const Dtype* gpu_diff();
  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();
  void update();
 private:
  void check_data();
  void check_diff();
  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> diff_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int count_;
};  // class Blob

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

#endif  // CAFFEINE_BLOB_HPP_