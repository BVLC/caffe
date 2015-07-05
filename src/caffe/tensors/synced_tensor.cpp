#include <climits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/tensor.hpp"

namespace caffe {

template<typename Dtype>
SyncedTensor<Dtype>::SyncedTensor()
    : Tensor<Dtype>(),
      cpu_tensor_(new CPUTensor<Dtype>()),
#ifndef CPU_ONLY
      gpu_tensor_(new GPUTensor<Dtype>()),
#endif  // #ifndef CPU_ONLY
      head_(UNINITIALIZED),
      own_cpu_data_(false) {
  this->type_ = SYNCED_TENSOR;
}

template<typename Dtype>
SyncedTensor<Dtype>::SyncedTensor(const vector<int>& shape)
    : Tensor<Dtype>(shape),
      cpu_tensor_(new CPUTensor<Dtype>()),
#ifndef CPU_ONLY
      gpu_tensor_(new GPUTensor<Dtype>()),
#endif  // #ifndef CPU_ONLY
      head_(UNINITIALIZED),
      own_cpu_data_(false) {
  this->type_ = SYNCED_TENSOR;
}

template<typename Dtype>
SyncedTensor<Dtype>::SyncedTensor(const int dim0, const int dim1,
    const int dim2, const int dim3)
    : Tensor<Dtype>(dim0, dim1, dim2, dim3),
      cpu_tensor_(new CPUTensor<Dtype>()),
#ifndef CPU_ONLY
      gpu_tensor_(new GPUTensor<Dtype>()),
#endif  // #ifndef CPU_ONLY
      head_(UNINITIALIZED),
      own_cpu_data_(false) {
  this->type_ = SYNCED_TENSOR;
}

template<typename Dtype>
SyncedTensor<Dtype>::~SyncedTensor() {
}

template<typename Dtype>
void SyncedTensor<Dtype>::FromProto(const TensorProto& proto, bool reshape) {
  CHECK(cpu_tensor_);
  cpu_tensor_->FromProto(proto, reshape);
}

template<typename Dtype>
void SyncedTensor<Dtype>::ToProto(TensorProto* proto) {
  to_cpu();
  if (cpu_tensor_) {
    cpu_tensor_->ToProto(proto);
  }
}

template<typename Dtype>
void SyncedTensor<Dtype>::malloc(const size_t num) {
  if (current_tensor_) {
    current_tensor()->malloc(num);
  }
  Tensor<Dtype>::malloc(num);
}

template<typename Dtype>
void SyncedTensor<Dtype>::free() {
  if (this->own_data_) {
    if (cpu_tensor_) {
      cpu_tensor_->free();
    }
#ifndef CPU_ONLY
    if (gpu_tensor_) {
      gpu_tensor_->free();
    }
#endif  // #ifndef CPU_ONLY
    Tensor<Dtype>::free();
  }
}

template<typename Dtype>
void SyncedTensor<Dtype>::choose_device() {
  switch (Caffe::mode()) {
  case Caffe::GPU:
    to_gpu();
    break;
  case Caffe::CPU:
  default:
    to_cpu();
    break;
  }
}

template<typename Dtype>
void SyncedTensor<Dtype>::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    cpu_tensor_->malloc(this->size_);
    cpu_tensor_->mem_set(0);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    current_tensor_ = cpu_tensor_;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_tensor_->ptr() == NULL) {
      cpu_tensor_->malloc(this->size_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(this->size_, gpu_tensor_->ptr(),
                     cpu_tensor_->mutable_ptr());
    head_ = SYNCED;
    current_tensor_ = cpu_tensor_;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

template<typename Dtype>
void SyncedTensor<Dtype>::to_gpu() {
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    gpu_tensor_->malloc(this->size_);
    gpu_tensor_->mem_set(0);
    head_ = HEAD_AT_GPU;
    current_tensor_ = gpu_tensor_;
    break;
  case HEAD_AT_CPU:
    if (gpu_tensor_->ptr() == NULL) {
      gpu_tensor_->malloc(this->size_);
    }
    caffe_gpu_memcpy(this->size_, cpu_tensor_->ptr(),
                     gpu_tensor_->mutable_ptr());
    head_ = SYNCED;
    current_tensor_ = gpu_tensor_;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

template<typename Dtype>
const Dtype* SyncedTensor<Dtype>::cpu_data() {
  to_cpu();
  return cpu_tensor_->data();
}

template<typename Dtype>
Dtype* SyncedTensor<Dtype>::mutable_cpu_data() {
  to_cpu();
  return cpu_tensor_->mutable_data();
}

template<typename Dtype>
void SyncedTensor<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  CHECK(cpu_tensor_);
  if (own_cpu_data_) {
    cpu_tensor_->free();
  }
  cpu_tensor_->set_data(data);
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

template<typename Dtype>
const Dtype* SyncedTensor<Dtype>::gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  return gpu_tensor_->data();
#endif  // #ifndef CPU_ONLY
  return NULL;
}

template<typename Dtype>
Dtype* SyncedTensor<Dtype>::mutable_gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  return gpu_tensor_->mutable_data();
#endif  // #ifndef CPU_ONLY
  return NULL;
}

template<typename Dtype>
void SyncedTensor<Dtype>::mem_set(const Dtype value) {
  current_tensor()->mem_set(value);
}

template<typename Dtype>
void SyncedTensor<Dtype>::mem_copy(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), SYNCED_TENSOR);
  current_tensor()->mem_copy(const_cast<Tensor<Dtype>&>(that).tensor());
}

template<typename Dtype>
void SyncedTensor<Dtype>::abs(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), SYNCED_TENSOR);
  current_tensor()->abs(const_cast<Tensor<Dtype>&>(that).tensor());
}

template<> void SyncedTensor<int>::pow(const Tensor<int>& that,
    const int value) {
  NOT_IMPLEMENTED;
}

template<> void SyncedTensor<unsigned int>::pow(
    const Tensor<unsigned int>& that, const unsigned int value) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void SyncedTensor<Dtype>::pow(const Tensor<Dtype>& that, const Dtype value) {
  CHECK_EQ(that.type(), SYNCED_TENSOR);
  current_tensor()->pow(const_cast<Tensor<Dtype>&>(that).tensor(), value);
}

template<> void SyncedTensor<int>::scale(int scale_factor) {
  NOT_IMPLEMENTED;
}

template<> void SyncedTensor<unsigned int>::scale(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void SyncedTensor<Dtype>::scale(Dtype scale_factor) {
  current_tensor()->scale(scale_factor);
}

template<> void SyncedTensor<int>::add(const int value) {
  NOT_IMPLEMENTED;
}

template<>
void SyncedTensor<unsigned int>::add(const unsigned int value) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void SyncedTensor<Dtype>::add(const Dtype value) {
  current_tensor()->add(value);
}

template<> void SyncedTensor<int>::add(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<>
void SyncedTensor<unsigned int>::add(const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void SyncedTensor<Dtype>::add(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), SYNCED_TENSOR);
  current_tensor()->add(const_cast<Tensor<Dtype>&>(that).tensor());
}

template<> void SyncedTensor<int>::add(const int value,
                                       const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<>
void SyncedTensor<unsigned int>::add(const unsigned int value,
    const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void SyncedTensor<Dtype>::add(const Dtype value, const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), SYNCED_TENSOR);
  current_tensor()->add(value, const_cast<Tensor<Dtype>&>(that).tensor());
}

template<> int SyncedTensor<int>::dot(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
  return 0;
}

template<>
unsigned int SyncedTensor<unsigned int>::dot(const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
  return 0;
}

template<typename Dtype>
Dtype SyncedTensor<Dtype>::dot(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), SYNCED_TENSOR);
  return current_tensor()->dot(const_cast<Tensor<Dtype>&>(that).tensor());
}

template<> void SyncedTensor<int>::mul(const int value) {
  NOT_IMPLEMENTED;
}

template<>
void SyncedTensor<unsigned int>::mul(const unsigned int value) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void SyncedTensor<Dtype>::mul(const Dtype value) {
  current_tensor()->mul(value);
}

template<> void SyncedTensor<int>::div(const int value) {
  NOT_IMPLEMENTED;
}

template<>
void SyncedTensor<unsigned int>::div(const unsigned int value) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void SyncedTensor<Dtype>::div(const Dtype value) {
  current_tensor()->div(value);
}

template<> void SyncedTensor<int>::cmul(const Tensor<int>& first,
    const Tensor<int>& second) {
  NOT_IMPLEMENTED;
}

template<>
void SyncedTensor<unsigned int>::cmul(const Tensor<unsigned int>& first,
    const Tensor<unsigned int>& second) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void SyncedTensor<Dtype>::cmul(const Tensor<Dtype>& first,
    const Tensor<Dtype>& second) {
  CHECK_EQ(first.type(), SYNCED_TENSOR);
  CHECK_EQ(second.type(), SYNCED_TENSOR);
  current_tensor()->cmul(const_cast<Tensor<Dtype>&>(first).tensor(),
                         const_cast<Tensor<Dtype>&>(second).tensor());
}

template<> void SyncedTensor<int>::cdiv(const Tensor<int>& first,
    const Tensor<int>& second) {
  NOT_IMPLEMENTED;
}

template<>
void SyncedTensor<unsigned int>::cdiv(const Tensor<unsigned int>& first,
    const Tensor<unsigned int>& second) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void SyncedTensor<Dtype>::cdiv(const Tensor<Dtype>& first,
    const Tensor<Dtype>& second) {
  CHECK_EQ(first.type(), SYNCED_TENSOR);
  CHECK_EQ(second.type(), SYNCED_TENSOR);
  current_tensor()->cdiv(const_cast<Tensor<Dtype>&>(first).tensor(),
                         const_cast<Tensor<Dtype>&>(second).tensor());
}

template<> void SyncedTensor<int>::mv(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<>
void SyncedTensor<unsigned int>::mv(const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void SyncedTensor<Dtype>::mv(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), SYNCED_TENSOR);
  current_tensor()->mv(const_cast<Tensor<Dtype>&>(that).tensor());
}

template<> void SyncedTensor<int>::addmv(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<>
void SyncedTensor<unsigned int>::addmv(const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void SyncedTensor<Dtype>::addmv(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), SYNCED_TENSOR);
  current_tensor()->addmv(const_cast<Tensor<Dtype>&>(that).tensor());
}

template<> void SyncedTensor<int>::mm(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<>
void SyncedTensor<unsigned int>::mm(const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void SyncedTensor<Dtype>::mm(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), SYNCED_TENSOR);
  current_tensor()->mm(const_cast<Tensor<Dtype>&>(that).tensor());
}

template<> void SyncedTensor<int>::addmm(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<> void SyncedTensor<unsigned int>::addmm(
    const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void SyncedTensor<Dtype>::addmm(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), SYNCED_TENSOR);
  current_tensor()->addmm(const_cast<Tensor<Dtype>&>(that).tensor());
}

template<> int SyncedTensor<int>::asum() {
  NOT_IMPLEMENTED;
  return 0;
}

template<> unsigned int SyncedTensor<unsigned int>::asum() {
  NOT_IMPLEMENTED;
  return 0;
}

template<typename Dtype>
Dtype SyncedTensor<Dtype>::asum() {
  return current_tensor()->asum();
}

template<> int SyncedTensor<int>::sumsq() {
  NOT_IMPLEMENTED;
  return 0;
}

template<> unsigned int SyncedTensor<unsigned int>::sumsq() {
  NOT_IMPLEMENTED;
  return 0;
}

template<typename Dtype>
Dtype SyncedTensor<Dtype>::sumsq() {
  return current_tensor()->sumsq();
}

INSTANTIATE_CLASS(SyncedTensor);
template class SyncedTensor<int>;
template class SyncedTensor<unsigned int>;

}  // namespace caffe

