#include <climits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/tensor.hpp"
#include "caffe/util/math_functions.hpp"

#ifndef CPU_ONLY  //  CPU-only Caffe.

namespace caffe {

template<typename Dtype>
GPUTensor<Dtype>::GPUTensor()
    : Tensor<Dtype>() {
  this->type_ = GPU_TENSOR;
}

template<typename Dtype>
GPUTensor<Dtype>::GPUTensor(const vector<int>& shape)
    : Tensor<Dtype>(shape) {
  this->type_ = GPU_TENSOR;
}

template<typename Dtype>
GPUTensor<Dtype>::GPUTensor(const int dim0, const int dim1, const int dim2,
    const int dim3)
    : Tensor<Dtype>(dim0, dim1, dim2, dim3) {
  this->type_ = GPU_TENSOR;
}

template<typename Dtype>
GPUTensor<Dtype>::~GPUTensor() {
//  free();
}

template<typename Dtype>
void GPUTensor<Dtype>::FromProto(const TensorProto& proto, bool reshape) {
  Tensor<Dtype>::FromProto(proto, reshape);
  //  copy data
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUTensor<Dtype>::ToProto(TensorProto* proto) {
  Tensor<Dtype>::ToProto(proto);
  //  copy data
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUTensor<Dtype>::malloc(const size_t num) {
  CUDA_CHECK(cudaMalloc(&this->ptr_, num * sizeof(Dtype)));
  Tensor<Dtype>::malloc(num);
}

template<typename Dtype>
void GPUTensor<Dtype>::free() {
#ifndef CPU_ONLY
  if (this->own_data_ && this->ptr_ != NULL) {
    CUDA_CHECK(cudaFree(this->ptr_));
    Tensor<Dtype>::free();
  }
#endif  //  CPU_ONLY
}

template<typename Dtype>
void GPUTensor<Dtype>::mem_set(const Dtype value) {
  if (this->ptr_ != NULL) {
    caffe_gpu_memset(this->size_ * sizeof(Dtype), value, this->mutable_data());
  }
}

template<typename Dtype>
void GPUTensor<Dtype>::mem_copy(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), GPU_TENSOR);
  this->resize(that.size());
  if (that.size() > 0) {
    CHECK(that.data());
    CHECK(this->mutable_data());
    caffe_gpu_memcpy(this->size_ * sizeof(Dtype), that.data(),
        this->mutable_data());
  }
}

/*
 * Construction or extraction functions
 *

template<>
Dtype GPUTensor<int>::ones(Tensor* this, const vector<int> shape) {
  NOT_IMPLEMENTED;
}

template<>
Dtype GPUTensor<unsigned int>::ones(Tensor* this, const vector<int> shape) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
Dtype GPUTensor<Dtype>::ones(Tensor* this, const vector<int> shape) {
  this->Reshape(shape);
  caffe_set(this->size(), static_cast<Dtype>(1), this->mutable_data());
}

template<typename Dtype>
Dtype GPUTensor<Dtype>::ones(Tensor* this, const TensorShape& shape) {
  ones(shapeToVector(shape));
}

template<typename Dtype>
Dtype GPUTensor<Dtype>::ones(const int dim0, int dim1, int dim2, int dim3) {
  ones(dimToShape(dim0, dim1, dim2, dim3));
}

template<>
Dtype GPUTensor<int>::zeros(const vector<int> shape) {
  NOT_IMPLEMENTED;
}

template<>
Dtype GPUTensor<unsigned int>::zeros(const vector<int> shape) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
Dtype GPUTensor<Dtype>::zeros(const vector<int> shape) {
  this->Reshape(shape);
  caffe_memset(this->size() * sizeof(Dtype), (Dtype) 0., this->mutable_data());
}

template<typename Dtype>
Dtype GPUTensor<Dtype>::zeros(const TensorShape& shape) {
  zeros(shapeToVector(shape));
}

template<typename Dtype>
Dtype GPUTensor<Dtype>::zeros(const int dim0, int dim1, int dim2, int dim3) {
  zeros(dimToShape(dim0, dim1, dim2, dim3));
}
*/

/*
 * Element-wise Mathematical Operations
 */
template<> void GPUTensor<int>::abs(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<> void GPUTensor<unsigned int>::abs(const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUTensor<Dtype>::abs(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), GPU_TENSOR);
  this->resize(this->size_);
  caffe_gpu_abs(this->size_, this->data(), this->mutable_data());
}

template<> void GPUTensor<int>::pow(const Tensor<int>& that, const int value) {
  NOT_IMPLEMENTED;
}

template<> void GPUTensor<unsigned int>::pow(const Tensor<unsigned int>& that,
    const unsigned int value) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUTensor<Dtype>::pow(const Tensor<Dtype>& that, const Dtype value) {
  CHECK_EQ(that.type(), GPU_TENSOR);
  this->resize(this->size_);
  caffe_gpu_powx(this->size_, this->data(), value, this->mutable_data());
}

template<> void GPUTensor<int>::scale(int scale_factor) {
  NOT_IMPLEMENTED;
}

template<> void GPUTensor<unsigned int>::scale(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUTensor<Dtype>::scale(Dtype scale_factor) {
  if (this->data() == NULL) {
    return;
  }
  caffe_gpu_scal(this->size_, scale_factor, this->mutable_data());
}

template<> void GPUTensor<int>::add(const int value) {
  NOT_IMPLEMENTED;
}

template<>
void GPUTensor<unsigned int>::add(const unsigned int value) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUTensor<Dtype>::add(const Dtype value) {
  caffe_gpu_add_scalar(this->size_, value, this->mutable_data());
}

template<> void GPUTensor<int>::add(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<>
void GPUTensor<unsigned int>::add(const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUTensor<Dtype>::add(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), GPU_TENSOR);
  CHECK(this->size_ <= that.size());
  caffe_gpu_add(this->size_, this->data(), that.data(), this->mutable_data());
}

template<> void GPUTensor<int>::add(const int value, const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<>
void GPUTensor<unsigned int>::add(const unsigned int value,
    const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUTensor<Dtype>::add(const Dtype value, const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), GPU_TENSOR);
  caffe_gpu_axpy(this->size_, value, that.data(), this->mutable_data());
}

template<> int GPUTensor<int>::dot(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
  return 0;
}

template<>
unsigned int GPUTensor<unsigned int>::dot(const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
  return 0;
}

template<typename Dtype>
Dtype GPUTensor<Dtype>::dot(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), GPU_TENSOR);
  CHECK(this->size_ <= that.size());
  Dtype result = 0;
  caffe_gpu_dot(this->size_, this->data(), that.data(), &result);
  return result;
}

template<> void GPUTensor<int>::mul(const int value) {
  NOT_IMPLEMENTED;
}

template<>
void GPUTensor<unsigned int>::mul(const unsigned int value) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUTensor<Dtype>::mul(const Dtype value) {
  caffe_gpu_axpy(this->size_, value, this->data(), this->mutable_data());
}

template<> void GPUTensor<int>::div(const int value) {
  NOT_IMPLEMENTED;
}

template<>
void GPUTensor<unsigned int>::div(const unsigned int value) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUTensor<Dtype>::div(const Dtype value) {
  CHECK_NE(0, value);
  mul(1 / value);
}

template<> void GPUTensor<int>::cmul(const Tensor<int>& first,
    const Tensor<int>& second) {
  NOT_IMPLEMENTED;
}

template<>
void GPUTensor<unsigned int>::cmul(const Tensor<unsigned int>& first,
                                   const Tensor<unsigned int>& second) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUTensor<Dtype>::cmul(const Tensor<Dtype>& first,
    const Tensor<Dtype>& second) {
  CHECK_EQ(first.type(), GPU_TENSOR);
  CHECK_EQ(second.type(), GPU_TENSOR);
  CHECK(first.data());
  CHECK(second.data());
  CHECK(this->mutable_data());
  CHECK_EQ(first.size(), second.size());
  this->resize(first.size());
  caffe_gpu_mul(this->size_, first.data(), second.data(), this->mutable_data());
}

template<> void GPUTensor<int>::cdiv(const Tensor<int>& first,
    const Tensor<int>& second) {
  NOT_IMPLEMENTED;
}

template<>
void GPUTensor<unsigned int>::cdiv(const Tensor<unsigned int>& first,
                                   const Tensor<unsigned int>& second) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUTensor<Dtype>::cdiv(const Tensor<Dtype>& first,
    const Tensor<Dtype>& second) {
  CHECK_EQ(first.type(), GPU_TENSOR);
  CHECK_EQ(second.type(), GPU_TENSOR);
  CHECK(first.data());
  CHECK(second.data());
  CHECK(this->mutable_data());
  CHECK_EQ(first.size(), second.size());
  this->resize(first.size());
  caffe_gpu_div(this->size_, first.data(), second.data(), this->mutable_data());
}

template<> void GPUTensor<int>::mv(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<>
void GPUTensor<unsigned int>::mv(const Tensor<unsigned int>& that) {
  CHECK_EQ(that.type(), GPU_TENSOR);
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUTensor<Dtype>::mv(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), GPU_TENSOR);
  CHECK(this->size_ <= that.size());
  caffe_gpu_gemv(this->get_transpose(), this->shape(0), this->shape(1),
                 static_cast<Dtype>(1), this->data(), that.data(), (Dtype) 0.,
                 this->mutable_data());
}

template<> void GPUTensor<int>::addmv(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<>
void GPUTensor<unsigned int>::addmv(const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUTensor<Dtype>::addmv(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), GPU_TENSOR);
  CHECK(this->size_ <= that.size());
  caffe_gpu_gemv(this->get_transpose(), this->shape(0), this->shape(1),
                 static_cast<Dtype>(1), this->data(), that.data(),
                 static_cast<Dtype>(1), this->mutable_data());
}

template<> void GPUTensor<int>::mm(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<>
void GPUTensor<unsigned int>::mm(const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUTensor<Dtype>::mm(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), GPU_TENSOR);
  CHECK(this->size_ <= that.size());
  caffe_gpu_gemm(this->get_transpose(), that.get_transpose(), this->shape(0),
                 that.shape(1), this->shape(1), static_cast<Dtype>(1),
                 this->data(), that.data(), (Dtype) 0., this->mutable_data());
}

template<> void GPUTensor<int>::addmm(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<> void GPUTensor<unsigned int>::addmm(
    const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUTensor<Dtype>::addmm(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), GPU_TENSOR);
  CHECK(this->size_ <= that.size());
  caffe_gpu_gemm(this->get_transpose(), that.get_transpose(), this->shape(0),
                 that.shape(1), this->shape(1), static_cast<Dtype>(1),
                 this->data(), that.data(), static_cast<Dtype>(1),
                 this->mutable_data());
}

template<> int GPUTensor<int>::asum() {
  NOT_IMPLEMENTED;
  return 0;
}

template<> unsigned int GPUTensor<unsigned int>::asum() {
  NOT_IMPLEMENTED;
  return 0;
}

template<typename Dtype>
Dtype GPUTensor<Dtype>::asum() {
  if (this->data() == NULL) {
    return 0;
  }
  Dtype result = 0;
  caffe_gpu_asum(this->size_, this->data(), &result);
  return result;
}

template<> int GPUTensor<int>::sumsq() {
  NOT_IMPLEMENTED;
  return 0;
}

template<> unsigned int GPUTensor<unsigned int>::sumsq() {
  NOT_IMPLEMENTED;
  return 0;
}

template<typename Dtype>
Dtype GPUTensor<Dtype>::sumsq() {
  if (this->data() == NULL) {
    return 0;
  }
  Dtype result = 0;
  caffe_gpu_dot(this->size_, this->data(), this->data(), &result);
  return result;
}

INSTANTIATE_CLASS(GPUTensor);
template class GPUTensor<int>;
template class GPUTensor<unsigned int>;

}  //  namespace caffe

#endif  // #ifndef CPU_ONLY
