#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "caffe/backend/device.hpp"
#include "caffe/util/string_and_path.hpp"

namespace caffe {

Device::Device() {
  backend_ = BACKEND_CPU;
}

void Device::Init() {
#ifdef USE_SQLITE
  database_ = make_shared<SQLiteHelper>(
      sanitize_path_string(name()).substr(0, 32) + "." +
      backend_name(backend()));
  database_->CreateTables();
#endif
}

#ifdef USE_SQLITE
shared_ptr<SQLiteHelper> Device::get_database() {
  return database_;
}
#endif  // USE_SQLITE


string Device::name() { return "CPU"; }

void Device::MallocMemHost(uint_tp size, void** ptr) {
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

vptr<void> Device::MallocMemDevice(uint_tp size, void** ptr,
                                               bool zero_copy) {
  return vptr<void>();
}

void Device::FreeMemHost(void* ptr) {
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif  // USE_MKL
}

void Device::FreeMemDevice(vptr<void> ptr) {
}

uint_tp Device::num_queues() {
  return 1;
}

bool Device::is_host_unified() {
  return true;
}

bool Device::CheckVendor(string vendor) {
  return "CPU";
}

bool Device::CheckCapability(DeviceCapability cap) {
  return false;
}

bool Device::CheckZeroCopy(vptr<const void> gpu_ptr, void* cpu_ptr,
                                       uint_tp size) {
  NOT_IMPLEMENTED;
}

void Device::get_threads(const vector<size_t>* work_size,
                         vector<size_t>* group,
                         vector<size_t>* local,
                         DeviceKernel* kernel,
                         bool auto_select) {
  NOT_IMPLEMENTED;
}

bool Device::CheckType(string type) {
  if (type.compare("CPU") == 1)
    return true;
  return false;
}

void Device::SwitchQueue(uint_tp id) {}

void Device::FinishQueues() {}

shared_ptr<DeviceProgram> Device::CreateProgram() {
  return nullptr;
}

void Device::memcpy(const uint_tp n, vptr<const void> x, vptr<void> y) {
  NOT_IMPLEMENTED;
}

void Device::memcpy(const uint_tp n, const void* x, vptr<void> y) {
  NOT_IMPLEMENTED;
}

void Device::memcpy(const uint_tp n, vptr<const void> x, void* y) {
  NOT_IMPLEMENTED;
}

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

template<typename Dtype>
size_t Device::preferred_vector_width() {
  string type = safe_type_name<Dtype>();
  if (preferred_vector_widths_.find(type) != preferred_vector_widths_.end()) {
    return preferred_vector_widths_[type];
  } else {
    return 0;
  }
}

template size_t Device::preferred_vector_width<char>();
template size_t Device::preferred_vector_width<int8_t>();
template size_t Device::preferred_vector_width<uint8_t>();
template size_t Device::preferred_vector_width<int16_t>();
template size_t Device::preferred_vector_width<uint16_t>();
template size_t Device::preferred_vector_width<int32_t>();
template size_t Device::preferred_vector_width<uint32_t>();
template size_t Device::preferred_vector_width<int64_t>();
template size_t Device::preferred_vector_width<uint64_t>();
template size_t Device::preferred_vector_width<half_fp>();
template size_t Device::preferred_vector_width<float>();
template size_t Device::preferred_vector_width<double>();

bool Device::is_fast_unsafe_math() const {
  return fast_unsafe_math_;
}

uint_tp Device::current_queue_id() {
  return current_queue_id_;
}

uint_tp Device::memory_usage() {
  return memory_usage_;
}

uint_tp Device::peak_memory_usage() {
  return peak_memory_usage_;
#ifndef NDEBUG
  std::cout << "Peak memory usage: " << peak_memory_usage_ << std::endl;
#endif  // NDEBUG
}

void Device::increase_memory_usage(uint_tp bytes) {
  memory_usage_ += bytes;
  if (memory_usage_ > peak_memory_usage_) {
    peak_memory_usage_ = memory_usage_;
  }
#ifndef NDEBUG
  std::cout << "Current memory usage: " << memory_usage_ << std::endl;
  std::cout << "Peak memory usage: " << peak_memory_usage_ << std::endl;
#endif  // NDEBUG
}

void Device::decrease_memory_usage(uint_tp bytes) {
  memory_usage_ -= bytes;
}

void Device::reset_peak_memory_usage() {
  peak_memory_usage_ = memory_usage_;
}

template<typename Dtype>
shared_ptr<Blob<Dtype> > Device::Buffer(vector<int_tp> shape, int_tp* lock_id) {
  CHECK(lock_id);
  shared_ptr<Blob<Dtype> > blob = make_shared<Blob<Dtype> >(this);
  vector<int_tp> buffer_shape(1, safe_sizeof<Dtype>());
  for(size_t i = 0; i < shape.size(); ++i) {
    buffer_shape[0] *= shape[i];
  }
  CHECK_GT(buffer_shape[0], 0) << "Trying to create a device buffer of size 0.";

  // Classify buffers by their log-size to pick reasonable sized buffers from
  // the existing ones (or create a new one otherwise)
  size_t log_size = static_cast<size_t>(std::floor(
                    std::log(static_cast<double>(buffer_shape[0]))));
  size_t min_size = static_cast<size_t>(
                    std::exp(static_cast<double>(log_size)));
  size_t max_size = static_cast<size_t>(
                    std::exp(static_cast<double>(log_size + 1)));

  // Ensure the thread safety of this function
  buffer_vec_mutex_.lock();
  size_t buffer_id = -1;
  for (size_t i = 0; i < buffers_.size(); ++i) {
    // Do not use buffers that are disproportionally different in size
    if (buffers_[i]->byte_count() >= min_size &&
        buffers_[i]->byte_count() <= max_size) {
      bool expected = false;
      if (buffer_flags_[i]->compare_exchange_strong(expected, true)) {
        buffer_id = i;
        break;
      }
    }
  }

  // No buffers available, create a new one
  if (buffer_id == -1) {
    buffer_id = buffers_.size();
    buffers_.push_back(make_shared<Blob<uint8_t> >(this));
    buffer_flags_.push_back(std::make_shared<std::atomic<bool> >(true));
  }

  Blob<uint8_t>* buffer = buffers_[buffer_id].get();
  buffer_vec_mutex_.unlock();

  // Ensure the buffer is big enough for the request
  buffer->Reshape(buffer_shape);

  // Share data between returned Blob object and internal device buffer
  blob->Reshape(shape);  // Will not cause allocation (lazy allocation)
  blob->ShareDataBase(buffer);
  blob->ShareDiffBase(buffer);

  *lock_id = buffer_id;
  return blob;
}

template shared_ptr<Blob<half_fp> > Device::Buffer(vector<int_tp> shape,
                                                   int_tp* lock_id);
template shared_ptr<Blob<float> > Device::Buffer(vector<int_tp> shape,
                                                   int_tp* lock_id);
template shared_ptr<Blob<double> > Device::Buffer(vector<int_tp> shape,
                                                   int_tp* lock_id);
template shared_ptr<Blob<int8_t> > Device::Buffer(vector<int_tp> shape,
                                                   int_tp* lock_id);
template shared_ptr<Blob<int16_t> > Device::Buffer(vector<int_tp> shape,
                                                   int_tp* lock_id);
template shared_ptr<Blob<int32_t> > Device::Buffer(vector<int_tp> shape,
                                                   int_tp* lock_id);
template shared_ptr<Blob<int64_t> > Device::Buffer(vector<int_tp> shape,
                                                   int_tp* lock_id);
template shared_ptr<Blob<uint8_t> > Device::Buffer(vector<int_tp> shape,
                                                   int_tp* lock_id);
template shared_ptr<Blob<uint16_t> > Device::Buffer(vector<int_tp> shape,
                                                   int_tp* lock_id);
template shared_ptr<Blob<uint32_t> > Device::Buffer(vector<int_tp> shape,
                                                   int_tp* lock_id);
template shared_ptr<Blob<uint64_t> > Device::Buffer(vector<int_tp> shape,
                                                   int_tp* lock_id);

void Device::unlock_buffer(int_tp* lock_id) {
  buffer_vec_mutex_.lock();
  if (*lock_id < buffer_flags_.size() && lock_id > 0) {
    bool expected = true;
    while (!(buffer_flags_[*lock_id]->
             compare_exchange_weak(expected, false))) { }
  }
  *lock_id = -1;
  buffer_vec_mutex_.unlock();
}

}  // namespace caffe
