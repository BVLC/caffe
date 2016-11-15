#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/sparse_blob.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void SparseBlob<Dtype>::Reshape(const vector<int>& shape, const int nnz) {
  CHECK_EQ(shape.size(), 2);
  CHECK_GE(shape[0], 0);
  CHECK_GE(shape[1], 0);
  CHECK_GE(nnz, 0);

  int previous_num = 0;
  if (this->shape_.size() > 0) {
    previous_num = this->shape_[0];
  }
  this->shape_.resize(2);
  this->shape_[0] = shape[0];
  this->shape_[1] = shape[1];
  this->count_ = shape[0] * shape[1];
  if (this->count_) {
    //std::cerr << "nnz=" << nnz << " / nnz_=" << nnz_ << std::endl;
    if (nnz != nnz_) {
      nnz_ = nnz;
      this->data_.reset(new SyncedMemory(nnz_ * sizeof(Dtype)));
      indices_.reset(new SyncedMemory(nnz_ * sizeof(int)));
    }
    if (previous_num != shape[0]) {
      ptr_.reset(new SyncedMemory((this->shape_[0] + 1) * sizeof(int)));
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
  vector<int> shape(2);
  shape[0] = num;
  shape[1] = channels;
  Reshape(shape, 1);
}

template<typename Dtype>
void SparseBlob<Dtype>::Reshape(const vector<int>& shape) {
  Reshape(shape, 1);
}

template<typename Dtype>
void SparseBlob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  if (const SparseBlob<Dtype>* sparseBlob =
      dynamic_cast<SparseBlob<Dtype>*>((Blob<Dtype>*) (&other))) {
    Reshape(other.shape(), sparseBlob->nnz());
  } else {
    Reshape(other.shape());
  }
}

template<typename Dtype>
SparseBlob<Dtype>::SparseBlob(const vector<int>& shape,
                              const int nnz)
  :nnz_(0) {
  Reshape(shape, nnz);
}

template<typename Dtype>
SparseBlob<Dtype>::SparseBlob(const int num, const int channels, const int nnz)
  :nnz_(0) {
  vector<int> shape(2);
  shape[0] = num;
  shape[1] = channels;
  Reshape(shape, nnz);
}

template<typename Dtype>
const int* SparseBlob<Dtype>::gpu_shape() const {
  LOG(FATAL)<< "gpu_shape is not supported";
  return 0;
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
                                     int nnz, int total_size) {
  CHECK(data);
  CHECK(indices);
  CHECK(ptr);
  nnz_ = nnz;
  if (total_size == -1) {
    total_size = nnz;
  }
  CHECK_GE(total_size, nnz);
  this->data_->set_cpu_data(reinterpret_cast<void*>(data),
                            total_size * sizeof(Dtype));
  indices_->set_cpu_data(reinterpret_cast<void*>(indices),
                         total_size * sizeof(int));
  ptr_->set_cpu_data(reinterpret_cast<void*>(ptr),
		     (this->shape_[0] + 1) * sizeof(int));
}
template<typename Dtype>
void SparseBlob<Dtype>::set_gpu_data(Dtype* data, int* indices, int* ptr,
                                     int nnz, int total_size) {
  CHECK(data);
  CHECK(indices);
  CHECK(ptr);
  nnz_ = nnz;
  if (total_size == -1) {
    total_size = nnz;
  }
  CHECK_GE(total_size, nnz);
  this->data_->set_gpu_data(data, total_size * sizeof(Dtype));
  indices_->set_gpu_data(indices, total_size * sizeof(int));
  ptr_->set_gpu_data(ptr, (this->shape_[0] + 1) * sizeof(int));
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

template <typename Dtype>
Dtype SparseBlob<Dtype>::asum_data() const {
  CAFFE_NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype SparseBlob<Dtype>::asum_diff() const {
  CAFFE_NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype SparseBlob<Dtype>::sumsq_data() const {
  CAFFE_NOT_IMPLEMENTED;
    return 0;
}

template <typename Dtype>
Dtype SparseBlob<Dtype>::sumsq_diff() const {
  CAFFE_NOT_IMPLEMENTED;
    return 0;
}

template <typename Dtype>
void SparseBlob<Dtype>::scale_data(Dtype scale_factor) {
  CAFFE_NOT_IMPLEMENTED;
}

template <typename Dtype>
void SparseBlob<Dtype>::scale_diff(Dtype scale_factor) {
  CAFFE_NOT_IMPLEMENTED;
}

template<typename Dtype>
void SparseBlob<Dtype>::CopyFrom(const Blob<Dtype>& source, bool copy_diff,
                                 bool reshape) {
  LOG(FATAL)<< "CopyFrom is not supported";
}

template<typename Dtype>
void SparseBlob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
  LOG(FATAL)<< "FromProto is not supported";
  return;
}

template<typename Dtype>
void SparseBlob<Dtype>::ToProto(BlobProto* proto, bool write_diff) const {
  LOG(FATAL)<< "ToProto is not supported";
  return;
}

INSTANTIATE_CLASS(SparseBlob);
template class SparseBlob<int>;
template class SparseBlob<unsigned int>;

}  // namespace caffe

