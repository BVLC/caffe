#include "caffe/sparse_blob.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void SparseBlob<Dtype>::Reshape(const int num, const int channels,
                                const int nzz) {
  CHECK_GE(num, 0);
  CHECK_GE(channels, 0);
  CHECK_GE(nzz, 0);

  const int previous_num = this->num_;
  this->num_ = num;
  this->channels_ = channels;
  this->height_ = 1;
  this->width_ = 1;
  this->count_ = this->num_ * this->channels_;
  if (this->count_) {
    if (nzz != nzz_) {
      nzz_ = nzz;
      this->data_.reset(new SyncedMemory(nzz_ * sizeof(Dtype)));
      indices_.reset(new SyncedMemory(nzz_ * sizeof(int)));
    }
    if (previous_num != num) {
      ptr_.reset(new SyncedMemory((this->num_ + 1) * sizeof(int)));
    }
  } else {
    this->data_.reset(reinterpret_cast<SyncedMemory*>(NULL));
    indices_.reset(reinterpret_cast<SyncedMemory*>(NULL));
    ptr_.reset(reinterpret_cast<SyncedMemory*>(NULL));
  }
}

template<typename Dtype>
void SparseBlob<Dtype>::Reshape(const int num, const int channels,
                                const int height, const int width) {
  CHECK_EQ(height, 1);
  CHECK_EQ(width, 1);
  Reshape(num, channels, 1);  // 1 to make sure something is created
}

template<typename Dtype>
void SparseBlob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  if (const SparseBlob<Dtype>* sparseBlob =
      dynamic_cast<SparseBlob<Dtype>*>((Blob<Dtype>*) (&other))) {
    Reshape(sparseBlob->num(), sparseBlob->channels(), sparseBlob->nzz());
  } else {
    Reshape(other.num(), other.channels(), other.height(), other.width());
  }
}

template<typename Dtype>
SparseBlob<Dtype>::SparseBlob(const int num, const int channels,
                              const int nzz) {
  nzz_ = 0;
  this->num_ = 0;
  Reshape(num, channels, nzz);
}

template<typename Dtype>
void SparseBlob<Dtype>::set_cpu_data(Dtype* data) {
  LOG(FATAL)<< "set_cpu_data is not supported";
}

template<typename Dtype>
void SparseBlob<Dtype>::set_gpu_data(Dtype* data) {
  LOG(FATAL)<< "set_gpu_data is not supported";
}

template<typename Dtype>
void SparseBlob<Dtype>::set_cpu_data(Dtype* data, int* indices, int* ptr,
                                     int nzz, int total_size) {
  CHECK(data);
  CHECK(indices);
  CHECK(ptr);
  nzz_ = nzz;
  if (total_size == -1) {
    total_size = nzz;
  }
  CHECK_GE(total_size, nzz);
  this->data_->set_cpu_data(reinterpret_cast<void*>(data),
                            total_size * sizeof(Dtype));
  indices_->set_cpu_data(reinterpret_cast<void*>(indices),
                         total_size * sizeof(int));
  ptr_->set_cpu_data(reinterpret_cast<void*>(ptr),
                     (this->num_ + 1) * sizeof(int));
}
template<typename Dtype>
void SparseBlob<Dtype>::set_gpu_data(Dtype* data, int* indices, int* ptr,
                                     int nzz, int total_size) {
  CHECK(data);
  CHECK(indices);
  CHECK(ptr);
  nzz_ = nzz;
  if (total_size == -1) {
    total_size = nzz;
  }
  CHECK_GE(total_size, nzz);
  this->data_->set_gpu_data(data, total_size * sizeof(Dtype));
  indices_->set_gpu_data(indices, total_size * sizeof(int));
  ptr_->set_gpu_data(ptr, (this->num_ + 1) * sizeof(int));
}

template<typename Dtype>
const Dtype* SparseBlob<Dtype>::cpu_diff() const {
  LOG(FATAL)<< "cpu_diff is not supported";
  return NULL;
}

template<typename Dtype>
const Dtype* SparseBlob<Dtype>::gpu_diff() const {
  LOG(FATAL)<< "gpu_diff is not supported";
  return NULL;
}

template<typename Dtype>
Dtype* SparseBlob<Dtype>::mutable_cpu_diff() {
  LOG(FATAL)<< "cpu_mutable_diff is not supported";
  return NULL;
}

template<typename Dtype>
Dtype* SparseBlob<Dtype>::mutable_gpu_diff() {
  LOG(FATAL)<< "gpu_mutable_diff is not supported";
  return NULL;
}

template<typename Dtype>
const int* SparseBlob<Dtype>::cpu_indices() const {
  CHECK(indices_);
  return (const int*) indices_->cpu_data();
}

template<typename Dtype>
const int* SparseBlob<Dtype>::cpu_ptr() const {
  CHECK(ptr_);
  return (const int*) ptr_->cpu_data();
}

template<typename Dtype>
const int* SparseBlob<Dtype>::gpu_indices() const {
  CHECK(indices_);
  return (const int*) indices_->gpu_data();
}

template<typename Dtype>
const int* SparseBlob<Dtype>::gpu_ptr() const {
  CHECK(ptr_);
  return (const int*) ptr_->gpu_data();
}

template<typename Dtype>
int* SparseBlob<Dtype>::mutable_cpu_indices() {
  CHECK(indices_);
  return reinterpret_cast<int*>(indices_->mutable_cpu_data());
}

template<typename Dtype>
int* SparseBlob<Dtype>::mutable_cpu_ptr() {
  CHECK(ptr_);
  return reinterpret_cast<int*>(ptr_->mutable_cpu_data());
}

template<typename Dtype>
int* SparseBlob<Dtype>::mutable_gpu_indices() {
  CHECK(indices_);
  return reinterpret_cast<int*>(indices_->mutable_gpu_data());
}

template<typename Dtype>
int* SparseBlob<Dtype>::mutable_gpu_ptr() {
  CHECK(ptr_);
  return reinterpret_cast<int*>(ptr_->mutable_gpu_data());
}

template<typename Dtype>
void SparseBlob<Dtype>::ShareData(const Blob<Dtype>& other) {
  LOG(FATAL)<< "ShareData is not supported";
}

template<typename Dtype>
void SparseBlob<Dtype>::ShareDiff(const Blob<Dtype>& other) {
  LOG(FATAL)<< "ShareDiff is not supported";
}

template<typename Dtype>
void SparseBlob<Dtype>::Update() {
  LOG(FATAL)<< "Update is not supported";
}

template<typename Dtype>
void SparseBlob<Dtype>::CopyFrom(const Blob<Dtype>& source, bool copy_diff,
                                 bool reshape) {
  LOG(FATAL)<< "CopyFrom is not supported";
}

template<typename Dtype>
void SparseBlob<Dtype>::FromProto(const BlobProto& proto) {
  LOG(FATAL)<< "FromProto is not supported";
}

template<typename Dtype>
void SparseBlob<Dtype>::ToProto(BlobProto* proto, bool write_diff) const {
  LOG(FATAL)<< "ToProto is not supported";
}
INSTANTIATE_CLASS(SparseBlob);
}  // namespace caffe

