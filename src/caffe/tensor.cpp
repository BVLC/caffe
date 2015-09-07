#include <climits>
#include <vector>

#include "caffe/syncedmem.hpp"
#include "caffe/tensor.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Tensor<Dtype>::Reshape(const vector<int>& shape) {
  ASSERT(shape.size() <= kMaxTensorAxes, "");
  count_ = 1;
  shape_.resize(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    ASSERT(shape[i] > 0, "");
    ASSERT(shape[i] <= INT_MAX / count_, "blob size exceeds INT_MAX");
    count_ *= shape[i];
    shape_[i] = shape[i];
  }
  if (count_ > capacity_) {
    // WARNING: Other tensors sharing data must be shared again.
    capacity_ = count_;
    mem_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
}

template <typename Dtype>
void Tensor<Dtype>::ReshapeLike(const Tensor<Dtype>& other) {
  Reshape(other.shape());
}

template <typename Dtype>
Tensor<Dtype>::Tensor(const vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(shape);
}

template <typename Dtype>
const Dtype* Tensor<Dtype>::cpu_mem() const {
  ASSERT(mem_, "");
  return (const Dtype*)mem_->cpu_data();
}

template <typename Dtype>
void Tensor<Dtype>::set_cpu_mem(Dtype* data) {
  ASSERT(data, "");
  mem_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Tensor<Dtype>::gpu_mem() const {
  ASSERT(mem_, "");
  return (const Dtype*)mem_->gpu_data();
}

template <typename Dtype>
Dtype* Tensor<Dtype>::mutable_cpu_mem() {
  ASSERT(mem_, "");
  return static_cast<Dtype*>(mem_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Tensor<Dtype>::mutable_gpu_mem() {
  ASSERT(mem_, "");
  return static_cast<Dtype*>(mem_->mutable_gpu_data());
}

template <typename Dtype>
void Tensor<Dtype>::ShareMem(const Tensor& other) {
  ASSERT(count_ == other.count(), "");
  mem_ = other.mem();
}

template <> unsigned int Tensor<unsigned int>::asum() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Tensor<int>::asum() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Tensor<Dtype>::asum() const {
  if (!mem_) { return 0; }
  switch (mem_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_mem());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_mem(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << mem_->head();
  }
  return 0;
}

template <> unsigned int Tensor<unsigned int>::sumsq() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Tensor<int>::sumsq() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Tensor<Dtype>::sumsq() const {
  Dtype sumsq;
  const Dtype* data;
  if (!mem_) { return 0; }
  switch (mem_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = cpu_mem();
    sumsq = caffe_cpu_dot(count_, data, data);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = gpu_mem();
    caffe_gpu_dot(count_, data, data, &sumsq);
#else
    NO_GPU;
#endif
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << mem_->head();
  }
  return sumsq;
}

template <> void Tensor<unsigned int>::scale(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Tensor<int>::scale(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Tensor<Dtype>::scale(Dtype scale_factor) {
  Dtype* data;
  if (!mem_) { return; }
  switch (mem_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = mutable_cpu_mem();
    caffe_scal(count_, scale_factor, data);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = mutable_gpu_mem();
    caffe_gpu_scal(count_, scale_factor, data);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << mem_->head();
  }
}

template <typename Dtype>
void Tensor<Dtype>::CopyChunkFrom(const Tensor& source, int count,
    int this_offset, int other_offset) {
  ASSERT(source.count() >= count + other_offset,
    "Chunk exceeds source memory: "
    << count << " + " << other_offset << " > " << source.count());
  ASSERT(this->count() >= count + this_offset, "Chunk exceeds target memory: "
    << count << " + " << this_offset << " > " << this->count());

  switch (mode()) {
  case Caffe::CPU:
    caffe_copy(count, source.cpu_mem() + other_offset,
        mutable_cpu_mem() + this_offset);
    break;
  case Caffe::GPU:
    caffe_copy(count, source.gpu_mem() + other_offset,
        mutable_gpu_mem() + this_offset);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Tensor<Dtype>::CopyFrom(const Tensor& source) {
  if (source.count() != count_ || source.shape() != shape_) {
    ASSERT(false, "Trying to copy blobs of different sizes.");
  }
  switch (mode()) {
  case Caffe::CPU:
    caffe_copy(count_, source.cpu_mem(),
        mutable_cpu_mem());
    break;
  case Caffe::GPU:
    caffe_copy(count_, source.gpu_mem(),
        static_cast<Dtype*>(mem_->mutable_gpu_data()));
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <> void Tensor<int>::SetValues(int value) {
  NOT_IMPLEMENTED;
  return;
}

template <> void Tensor<unsigned int>::SetValues(unsigned int value) {
  NOT_IMPLEMENTED;
  return;
}

template <typename Dtype>
void Tensor<Dtype>::SetValues(const Dtype value) {
  switch (mode()) {
  case Caffe::CPU:
    caffe_set(this->count(), value,
        this->mutable_cpu_mem());
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    caffe_gpu_set(this->count(), value,
        this->mutable_gpu_mem());
#else
    NO_GPU;
#endif
    break;
  default:
    ASSERT(false, "Unknown caffe mode.");
  }
}

template <> void Tensor<int>::MulFrom(const Tensor& source) {
  NOT_IMPLEMENTED;
  return;
}

template <> void Tensor<unsigned int>::MulFrom(const Tensor& source) {
  NOT_IMPLEMENTED;
  return;
}

template <typename Dtype>
void Tensor<Dtype>::MulFrom(const Tensor& source) {
  if (source.count() != count_ || source.shape() != shape_) {
    ASSERT(false, "Trying to add blobs of different sizes: "
      << source.count() << " != " << count_);
  }
  switch (mode()) {
  case Caffe::CPU:
    caffe_mul(count_, source.cpu_mem(),
        this->cpu_mem(),
        this->mutable_cpu_mem());
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    caffe_gpu_mul(count_, source.gpu_mem(),
        this->gpu_mem(),
        this->mutable_gpu_mem());
#else
    NO_GPU;
#endif
    break;
  default:
    ASSERT(false, "Unknown caffe mode.");
  }
}

template <> void Tensor<int>::AddFrom(const Tensor& source) {
  NOT_IMPLEMENTED;
  return;
}

template <> void Tensor<unsigned int>::AddFrom(const Tensor& source) {
  NOT_IMPLEMENTED;
  return;
}

template <typename Dtype>
void Tensor<Dtype>::AddFrom(const Tensor& source) {
  if (source.count() != count_ || source.shape() != shape_) {
    ASSERT(false, "Trying to add blobs of different sizes: "
      << source.count() << " != " << count_);
  }
  switch (mode()) {
  case Caffe::CPU:
    caffe_add(count_, source.cpu_mem(),
        this->cpu_mem(),
        this->mutable_cpu_mem());
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    caffe_gpu_add(count_, source.gpu_mem(),
        this->gpu_mem(),
        this->mutable_gpu_mem());
#else
    NO_GPU;
#endif
    break;
  default:
    ASSERT(false, "Unknown caffe mode.");
  }
}

// NOLINT_NEXT_LINE(runtime/int)
template <> void Tensor<int>::AddFromGPUPointer(int* ptr, long long size) {
  NOT_IMPLEMENTED;
  return;
}

template <> void Tensor<unsigned int>::AddFromGPUPointer(unsigned int* ptr,
// NOLINT_NEXT_LINE(runtime/int)
    long long size) {
  NOT_IMPLEMENTED;
  return;
}

template <typename Dtype>
// NOLINT_NEXT_LINE(runtime/int)
void Tensor<Dtype>::AddFromGPUPointer(Dtype* ptr, long long size) {
  if (size != count_) {
    ASSERT(false, "Trying to add blobs of different sizes: "
      << size << " != " << count_);
  }
#ifndef CPU_ONLY
  caffe_gpu_add(count_, ptr,
      this->gpu_mem(),
      this->mutable_gpu_mem());
#else
  ASSERT(false, "Operation not supported in CPU Only mode");
#endif
}

template <> void Tensor<int>::AddMulFrom(
    const Tensor& source, int alpha) {
  NOT_IMPLEMENTED;
  return;
}

template <> void Tensor<unsigned int>::AddMulFrom(
    const Tensor& source, unsigned int alpha) {
  NOT_IMPLEMENTED;
  return;
}

template <typename Dtype>
void Tensor<Dtype>::AddMulFrom(const Tensor& source, Dtype alpha) {
  if (source.count() != count_ || source.shape() != shape_) {
    ASSERT(false, "Trying to add blobs of different sizes: "
      << source.count() << " != " << count_);
  }
  switch (mode()) {
  case Caffe::CPU:
    caffe_axpy(count_, alpha,
        source.cpu_mem(),
        this->mutable_cpu_mem());
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    caffe_gpu_axpy(count_, alpha,
        source.gpu_mem(),
        this->mutable_gpu_mem());
#else
    NO_GPU;
#endif
    break;
  default:
    ASSERT(false, "Unknown caffe mode.");
  }
}

template <> void Tensor<int>::AddMulFromDynamicMode(
    const Tensor& source, int alpha) {
  NOT_IMPLEMENTED;
  return;
}

template <> void Tensor<unsigned int>::AddMulFromDynamicMode(
    const Tensor& source, unsigned int alpha) {
  NOT_IMPLEMENTED;
  return;
}

template <typename Dtype>
void Tensor<Dtype>::AddMulFromDynamicMode(const Tensor& source, Dtype alpha) {
  if (source.count() != count_ || source.shape() != shape_) {
    ASSERT(false, "Trying to add blobs of different sizes: "
      << source.count() << " != " << count_);
  }
  // We will perform update based on where the data is located.
  switch (mem_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(alpha),
        source.cpu_mem(),
        mutable_cpu_mem());
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    caffe_gpu_axpy<Dtype>(count_, Dtype(alpha),
        source.gpu_mem(),
        mutable_gpu_mem());
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <> int Tensor<int>::DotPFrom(const Tensor& source) {
  NOT_IMPLEMENTED;
  return 0;
}

template <> unsigned int Tensor<unsigned int>::DotPFrom(const Tensor& source) {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Tensor<Dtype>::DotPFrom(const Tensor& source) {
  if (source.count() != count_) {
    ASSERT(false, "Trying to dot blobs of different counts: "
      << source.count() << " != " << count_);
  }
  Dtype result;
  switch (mode()) {
  case Caffe::CPU:
    result = caffe_cpu_dot(count_,
        source.cpu_mem(),
        this->cpu_mem());
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    caffe_gpu_dot(count_,
        source.gpu_mem(),
        this->gpu_mem(),
        &result);
#else
    NO_GPU;
#endif
    break;
  default:
    ASSERT(false, "Unknown caffe mode.");
  }
  return result;
}


INSTANTIATE_CLASS(Tensor);
template class Tensor<int>;
template class Tensor<unsigned int>;

}  // namespace caffe

