#include <climits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/tensor.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
Tensor<Dtype>::Tensor()
    : own_data_(false),
      type_(UNKNOWN_TENSOR),
      ptr_(NULL),
      size_(0),
// capacity_ must be initialized before calling Reshape
      capacity_(0),
      transpose_(CblasNoTrans) {
}

template<typename Dtype>
Tensor<Dtype>::Tensor(const vector<int>& shape)
    : own_data_(false),
      type_(UNKNOWN_TENSOR),
      ptr_(NULL),
      size_(0),
      // capacity_ must be initialized before calling Reshape
      capacity_(0),
      transpose_(CblasNoTrans) {
  Reshape(shape);
}

template<typename Dtype>
Tensor<Dtype>::Tensor(const int dim0, const int dim1, const int dim2,
    const int dim3)
    : own_data_(false),
      type_(UNKNOWN_TENSOR),
      ptr_(NULL),
      size_(0),
      // capacity_ must be initialized before calling Reshape
      capacity_(0),
      transpose_(CblasNoTrans) {
  Reshape(dim0, dim1, dim2, dim3);
}

template<typename Dtype>
void Tensor<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxTensorAxes);
  size_ = 1;
  shape_.resize(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    CHECK_LE(shape[i], INT_MAX / size_) << "Tensor size exceeds INT_MAX";
    size_ *= shape[i];
    shape_[i] = shape[i];
  }
  resize(size_);
}

template<typename Dtype>
void Tensor<Dtype>::Reshape(const TensorShape& shape) {
  Reshape(shapeToVector(shape));
}

template<typename Dtype>
void Tensor<Dtype>::Reshape(const int dim0, const int dim1, const int dim2,
    const int dim3) {
  Reshape(dimToShape(dim0, dim1, dim2, dim3));
}

template<typename Dtype>
void Tensor<Dtype>::ReshapeLike(const Tensor<Dtype>& that) {
  Reshape(that.shape());
}

template<typename Dtype>
void Tensor<Dtype>::resize(const size_t size) {
  if (size > capacity_) {
    capacity_ = size;
    free();
    malloc(capacity_);
  }
  size_ = size;
}

template<typename Dtype>
void Tensor<Dtype>::CopyFrom(const Tensor<Dtype>& source, bool reshape) {
  if (source.size() != size_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  mem_copy(source);
}

template<typename Dtype>
void Tensor<Dtype>::FromProto(const TensorProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    shape.resize(proto.shape().dim_size());
    for (int i = 0; i < proto.shape().dim_size(); ++i) {
      shape[i] = proto.shape().dim(i);
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data is implemented in subclasses
}

template<typename Dtype>
void Tensor<Dtype>::ToProto(TensorProto* proto) {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  // copy data is implemented in subclasses
}

template<typename Dtype>
bool Tensor<Dtype>::ShapeEquals(const TensorProto& that) {
  vector<int> shape(that.shape().dim_size());
  for (int i = 0; i < that.shape().dim_size(); ++i) {
    shape[i] = that.shape().dim(i);
  }
  return shape_ == shape;
}

INSTANTIATE_CLASS(Tensor);
template class Tensor<int>;
template class Tensor<unsigned int>;

}  // namespace caffe

