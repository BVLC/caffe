#include <climits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/tensor.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
CPUTensor<Dtype>::CPUTensor()
    : Tensor<Dtype>() {
  this->type_ = CPU_TENSOR;
}

template<typename Dtype>
CPUTensor<Dtype>::CPUTensor(const vector<int>& shape)
    : Tensor<Dtype>(shape) {
  this->type_ = CPU_TENSOR;
}

template<typename Dtype>
CPUTensor<Dtype>::CPUTensor(const int dim0, const int dim1, const int dim2,
    const int dim3)
    : Tensor<Dtype>(dim0, dim1, dim2, dim3) {
  this->type_ = CPU_TENSOR;
}

template<typename Dtype>
CPUTensor<Dtype>::~CPUTensor() {
//  free();
}

template<typename Dtype>
void CPUTensor<Dtype>::FromProto(const TensorProto& proto, bool reshape) {
  Tensor<Dtype>::FromProto(proto, reshape);
  //  copy data
  Dtype* data_vec = this->mutable_data();
  for (int i = 0; i < this->size_; ++i) {
    data_vec[i] = proto.data(i);
  }
}

template<typename Dtype>
void CPUTensor<Dtype>::ToProto(TensorProto* proto) {
  Tensor<Dtype>::ToProto(proto);
  //  copy data
  proto->clear_data();
  const Dtype* data_vec = this->data();
  for (int i = 0; i < this->size_; ++i) {
    proto->add_data(data_vec[i]);
  }
}

template<typename Dtype>
void CPUTensor<Dtype>::malloc(const size_t num) {
  CaffeMallocHost(&(this->ptr_), num * sizeof(Dtype));
  Tensor<Dtype>::malloc(num);
}

template<typename Dtype>
void CPUTensor<Dtype>::free() {
  if (this->own_data_ && this->ptr_ != NULL) {
    CaffeFreeHost(this->ptr_);
    Tensor<Dtype>::free();
  }
}

template<typename Dtype>
void CPUTensor<Dtype>::set_data(Dtype* data) {
  CHECK(data);
  free();
  this->ptr_ = data;
  this->own_data_ = false;
}

/*
 * Construction or extraction functions
 */

template<typename Dtype>
void CPUTensor<Dtype>::mem_set(const Dtype value) {
  if (this->ptr_ != NULL) {
    caffe_memset(this->size_ * sizeof(Dtype), value, this->ptr_);
  }
}

template<typename Dtype>
void CPUTensor<Dtype>::mem_copy(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), CPU_TENSOR);
  this->resize(that.size());
  if (that.size() > 0) {
    CHECK(that.data());
    CHECK(this->mutable_data());
    caffe_copy(this->size_, that.data(), this->mutable_data());
  }
}

/*
 template<>
 Dtype CPUTensor<int>::ones(Tensor* result, const vector<int> shape) {
 NOT_IMPLEMENTED;
 }

 template<>
 Dtype CPUTensor<unsigned int>::ones(Tensor* result, const vector<int> shape) {
 NOT_IMPLEMENTED;
 }

 template<typename Dtype>
 Dtype CPUTensor<Dtype>::ones(Tensor* result, const vector<int> shape) {
 result->Reshape(shape);
 caffe_set(result->size(), static_cast<Dtype>(1), result->this->mutable_data());
 }

 template<typename Dtype>
 Dtype CPUTensor<Dtype>::ones(Tensor* result, const TensorShape& shape) {
 ones(shapeToVector(shape));
 }

 template<typename Dtype>
 Dtype CPUTensor<Dtype>::ones(const int dim0, int dim1, int dim2, int dim3) {
 ones(dimToShape(dim0, dim1, dim2, dim3));
 }

 template<>
 Dtype CPUTensor<int>::zeros(const vector<int> shape) {
 NOT_IMPLEMENTED;
 }

 template<>
 Dtype CPUTensor<unsigned int>::zeros(const vector<int> shape) {
 NOT_IMPLEMENTED;
 }

 template<typename Dtype>
 Dtype CPUTensor<Dtype>::zeros(const vector<int> shape) {
 result->Reshape(shape);
 caffe_memset(result->size() * sizeof(Dtype), (Dtype) 0.,
 result->this->mutable_data());
 }

 template<typename Dtype>
 Dtype CPUTensor<Dtype>::zeros(const TensorShape& shape) {
 zeros(shapeToVector(shape));
 }

 template<typename Dtype>
 Dtype CPUTensor<Dtype>::zeros(const int dim0, int dim1, int dim2, int dim3) {
 zeros(dimToShape(dim0, dim1, dim2, dim3));
 }
 */

/*
 * Element-wise Mathematical Operations
 */

template<>
void CPUTensor<int>::abs(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<>
void CPUTensor<unsigned int>::abs(const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void CPUTensor<Dtype>::abs(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), CPU_TENSOR);
  this->resize(that.size());
  if (that.size() > 0) {
    CHECK(that.data());
    CHECK(this->mutable_data());
    CHECK_GE(this->size_, that.size());
    caffe_abs(this->size_, that.data(), this->mutable_data());
  }
}

template<> void CPUTensor<int>::pow(const Tensor<int>& that, const int value) {
  NOT_IMPLEMENTED;
}

template<> void CPUTensor<unsigned int>::pow(const Tensor<unsigned int>& that,
    const unsigned int value) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void CPUTensor<Dtype>::pow(const Tensor<Dtype>& that, const Dtype value) {
  CHECK_EQ(that.type(), CPU_TENSOR);
  this->resize(that.size());
  if (that.size() > 0) {
    CHECK(that.data());
    CHECK(this->mutable_data());
    CHECK_GE(this->size_, that.size());
    caffe_powx(this->size_, that.data(), value, this->mutable_data());
  }
}

template<> void CPUTensor<int>::scale(int scale_factor) {
  NOT_IMPLEMENTED;
}

template<> void CPUTensor<unsigned int>::scale(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void CPUTensor<Dtype>::scale(Dtype scale_factor) {
  if (this->data() == NULL) {
    return;
  }
  caffe_scal(this->size_, scale_factor, this->mutable_data());
}

template<> void CPUTensor<int>::add(const int value) {
  NOT_IMPLEMENTED;
}

template<>
void CPUTensor<unsigned int>::add(const unsigned int value) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void CPUTensor<Dtype>::add(const Dtype value) {
  caffe_add_scalar(this->size_, value, this->mutable_data());
}

template<> void CPUTensor<int>::add(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<>
void CPUTensor<unsigned int>::add(const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void CPUTensor<Dtype>::add(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), CPU_TENSOR);
  this->resize(that.size());
  if (that.size() > 0) {
    CHECK(that.data());
    CHECK(this->mutable_data());
    CHECK_GE(this->size_, that.size());
    caffe_add(this->size_, this->data(), that.data(), this->mutable_data());
  }
}

template<> void CPUTensor<int>::add(const int value, const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<>
void CPUTensor<unsigned int>::add(const unsigned int value,
    const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void CPUTensor<Dtype>::add(const Dtype value, const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), CPU_TENSOR);
  this->resize(that.size());
  if (that.size() > 0) {
    CHECK(that.data());
    CHECK(this->mutable_data());
    CHECK_GE(this->size_, that.size());
    caffe_axpy<Dtype>(this->size_, value, that.data(), this->mutable_data());
  }
}

template<> int CPUTensor<int>::dot(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
  return 0;
}

template<>
unsigned int CPUTensor<unsigned int>::dot(const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
  return 0;
}

template<typename Dtype>
Dtype CPUTensor<Dtype>::dot(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), CPU_TENSOR);
  this->resize(that.size());
  if (that.size() > 0) {
    CHECK(that.data());
    CHECK(this->mutable_data());
    CHECK_GE(this->size_, that.size());
    return caffe_cpu_dot(this->size_, this->data(), that.data());
  }
  return 0;
}

template<> void CPUTensor<int>::mul(const int value) {
  NOT_IMPLEMENTED;
}

template<>
void CPUTensor<unsigned int>::mul(const unsigned int value) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void CPUTensor<Dtype>::mul(const Dtype value) {
  scale(value);
//  caffe_axpy(this->size_, value, this->data(), this->mutable_data());
}

template<> void CPUTensor<int>::div(const int value) {
  NOT_IMPLEMENTED;
}

template<>
void CPUTensor<unsigned int>::div(const unsigned int value) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void CPUTensor<Dtype>::div(const Dtype value) {
  CHECK_NE(0, value);
  mul(1 / value);
}

template<> void CPUTensor<int>::cmul(const Tensor<int>& first,
    const Tensor<int>& second) {
  NOT_IMPLEMENTED;
}

template<>
void CPUTensor<unsigned int>::cmul(const Tensor<unsigned int>& first,
    const Tensor<unsigned int>& second) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void CPUTensor<Dtype>::cmul(const Tensor<Dtype>& first,
    const Tensor<Dtype>& second) {
  CHECK_EQ(first.type(), CPU_TENSOR);
  CHECK_EQ(second.type(), CPU_TENSOR);
  this->resize(first.size());
  if (first.size() > 0) {
    CHECK_EQ(first.size(), second.size());
    CHECK(first.data());
    CHECK(second.data());
    CHECK(this->mutable_data());
    CHECK_GE(this->size_, first.size());
    caffe_mul(this->size_, first.data(), second.data(), this->mutable_data());
  }
}

template<> void CPUTensor<int>::cdiv(const Tensor<int>& first,
    const Tensor<int>& second) {
  NOT_IMPLEMENTED;
}

template<>
void CPUTensor<unsigned int>::cdiv(const Tensor<unsigned int>& first,
    const Tensor<unsigned int>& second) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void CPUTensor<Dtype>::cdiv(const Tensor<Dtype>& first,
    const Tensor<Dtype>& second) {
  CHECK_EQ(first.type(), CPU_TENSOR);
  CHECK_EQ(second.type(), CPU_TENSOR);
  this->resize(first.size());
  if (first.size() > 0) {
    CHECK_EQ(first.size(), second.size());
    CHECK(first.data());
    CHECK(second.data());
    CHECK(this->mutable_data());
    CHECK_GE(this->size_, first.size());
    caffe_div(this->size_, first.data(), second.data(), this->mutable_data());
  }
}

template<> void CPUTensor<int>::mv(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<>
void CPUTensor<unsigned int>::mv(const Tensor<unsigned int>& that) {
  CHECK_EQ(that.type(), CPU_TENSOR);
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void CPUTensor<Dtype>::mv(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), CPU_TENSOR);
  this->resize(that.size());
  if (that.size() > 0) {
    CHECK(that.data());
    CHECK(this->mutable_data());
    CHECK_GE(this->size_, that.size());
    caffe_cpu_gemv(this->get_transpose(), this->shape(0), this->shape(1),
                   static_cast<Dtype>(1), this->data(), that.data(), (Dtype) 0.,
                   this->mutable_data());
  }
}

template<> void CPUTensor<int>::addmv(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<>
void CPUTensor<unsigned int>::addmv(const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void CPUTensor<Dtype>::addmv(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), CPU_TENSOR);
  this->resize(that.size());
  if (that.size() > 0) {
    CHECK(that.data());
    CHECK(this->mutable_data());
    CHECK_GE(this->size_, that.size());
    caffe_cpu_gemv(this->get_transpose(), this->shape(0), this->shape(1),
                   static_cast<Dtype>(1), this->data(), that.data(),
                   static_cast<Dtype>(1), this->mutable_data());
  }
}

template<> void CPUTensor<int>::mm(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<>
void CPUTensor<unsigned int>::mm(const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void CPUTensor<Dtype>::mm(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), CPU_TENSOR);
  this->resize(that.size());
  if (that.size() > 0) {
    CHECK(that.data());
    CHECK(this->mutable_data());
    CHECK_GE(this->size_, that.size());
    caffe_cpu_gemm(this->get_transpose(), that.get_transpose(), this->shape(0),
                   that.shape(1), this->shape(1), static_cast<Dtype>(1),
                   this->data(), that.data(), (Dtype) 0., this->mutable_data());
  }
}

template<> void CPUTensor<int>::addmm(const Tensor<int>& that) {
  NOT_IMPLEMENTED;
}

template<> void CPUTensor<unsigned int>::addmm(
    const Tensor<unsigned int>& that) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void CPUTensor<Dtype>::addmm(const Tensor<Dtype>& that) {
  CHECK_EQ(that.type(), CPU_TENSOR);
  this->resize(that.size());
  if (that.size() > 0) {
    CHECK(that.data());
    CHECK(this->mutable_data());
    CHECK_GE(this->size_, that.size());
    caffe_cpu_gemm(this->get_transpose(), that.get_transpose(), this->shape(0),
                   that.shape(1), this->shape(1), static_cast<Dtype>(1),
                   this->data(), that.data(), static_cast<Dtype>(1),
                   this->mutable_data());
  }
}

template<> int CPUTensor<int>::asum() {
  NOT_IMPLEMENTED;
  return 0;
}

template<> unsigned int CPUTensor<unsigned int>::asum() {
  NOT_IMPLEMENTED;
  return 0;
}

template<typename Dtype>
Dtype CPUTensor<Dtype>::asum() {
  if (this->data() == NULL) {
    return 0;
  }
  return caffe_cpu_asum(this->size_, this->data());
}

template<> int CPUTensor<int>::sumsq() {
  NOT_IMPLEMENTED;
  return 0;
}

template<> unsigned int CPUTensor<unsigned int>::sumsq() {
  NOT_IMPLEMENTED;
  return 0;
}

template<typename Dtype>
Dtype CPUTensor<Dtype>::sumsq() {
  if (this->data() == NULL) {
    return 0;
  }
  return caffe_cpu_dot(this->size_, this->data(), this->data());
}

INSTANTIATE_CLASS(CPUTensor);
template class CPUTensor<int>;
template class CPUTensor<unsigned int>;

}  // namespace caffe

