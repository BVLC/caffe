#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "caffe/backend/device.hpp"

namespace caffe {

Device::Device() {}

void Device::Init() {}

string Device::name() { return "CPU"; }

void Device::MallocMemHost(void** ptr, uint_tp size) {
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size : 1, 64);
#else
#ifdef _MSC_VER
  *ptr = malloc(((size - 1) / CAFFE_MALLOC_CACHE_ALIGN + 1)
                * CAFFE_MALLOC_CACHE_ALIGN);
#else
  CHECK_EQ(0, posix_memalign(ptr, CAFFE_MALLOC_PAGE_ALIGN,
           ((size - 1) / CAFFE_MALLOC_CACHE_ALIGN + 1)
           * CAFFE_MALLOC_CACHE_ALIGN))
              << "Host memory allocation error of size: "
              << size << " b";
#endif  // _MSC_VER
#endif  // USE_MKL
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

void Device::FreeMemHost(void* ptr) {
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif  // USE_MKL
}

uint_tp Device::num_queues() {
  return 1;
}

bool Device::CheckVendor(string vendor) {
  return "CPU";
}

bool Device::CheckCapability(string cap) {
  return false;
}


bool Device::CheckType(string type) {
  if (type.compare("CPU") == 1)
    return true;
  return false;
}

void Device::SwitchQueue(uint_tp id) {}

void Device::FinishQueues() {}

void Device::memcpy(const uint_tp n, vptr<const void> x, vptr<void> y) {}

Backend Device::backend() const {
  return backend_;
}

uint_tp Device::id() const {
  return id_;
}

uint_tp Device::list_id() const {
  return list_id_;
}

size_t Device::workgroup_size(uint_tp id) {
  return max_local_sizes_[id % 3];
}

bool Device::is_fast_unsafe_math() const {
  return fast_unsafe_math_;
}

template<>
shared_ptr<Blob<half_float::half> > Device::Buffer(uint_tp id) {
  if (buff_h_.size() <= id) {
    shared_ptr<Blob<half_float::half> >
                                 blob_pointer(new Blob<half_float::half>(this));
    buff_h_.push_back(blob_pointer);
  }
  return buff_h_[id];
}

template<>
shared_ptr<Blob<float> > Device::Buffer(uint_tp id) {
  if (buff_f_.size() <= id) {
    shared_ptr<Blob<float> > blob_pointer(new Blob<float>(this));
    buff_f_.push_back(blob_pointer);
  }
  return buff_f_[id];
}

template<>
shared_ptr<Blob<double> > Device::Buffer(uint_tp id) {
  if (buff_d_.size() <= id) {
    shared_ptr<Blob<double> > blob_pointer(new Blob<double>(this));
    buff_d_.push_back(blob_pointer);
  }
  return buff_d_[id];
}

uint_tp Device::current_queue_id() {
  return current_queue_id_;
}

uint_tp Device::memory_usage() {
  return memory_usage_;
}

uint_tp Device::peak_memory_usage() {
  return peak_memory_usage_;
}

void Device::increase_memory_usage(uint_tp bytes) {
  memory_usage_ += bytes;
  if (memory_usage_ > peak_memory_usage_) {
    peak_memory_usage_ = memory_usage_;
  }
}

void Device::decrease_memory_usage(uint_tp bytes) {
  memory_usage_ -= bytes;
}

void Device::reset_peak_memory_usage() {
  peak_memory_usage_ = memory_usage_;
}


}  // namespace caffe
