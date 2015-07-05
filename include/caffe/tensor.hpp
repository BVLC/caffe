#ifndef CAFFE_TENSOR_HPP_
#define CAFFE_TENSOR_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

const int kMaxTensorAxes = INT_MAX;

namespace caffe {

inline vector<int> dimToShape(int dim0, int dim1, int dim2, int dim3) {
  vector<int> shape(4);
  shape[0] = dim0;
  shape[1] = dim1;
  shape[2] = dim2;
  shape[3] = dim3;
  return shape;
}

enum TensorType {
  UNKNOWN_TENSOR,
  CPU_TENSOR,
  GPU_TENSOR,
  SYNCED_TENSOR
};

/**
 * @brief A wrapper around void pointer serving as the view of the
 *        underlying memory storage. It's up to the users to specify the
 *        meanings of the data dimensions.
 *
 * TODO(dox): more thorough description.
 */
template<typename Dtype>
class Tensor {
 public:
  Tensor();

  explicit Tensor(const vector<int>& shape);

  /// @brief Convenience method for Tensor whose dimensions are equal to or
  ///        less than 4.
  explicit Tensor(const int dim0, const int dim1 = 1, const int dim2 = 1,
      const int dim3 = 1);

  virtual ~Tensor() {
  }

  TensorType type() const {
    return type_;
  }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  inline int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes()) << "axis " << axis_index
        << " out of range for " << num_axes() << "-D Tensor with shape "
        << shape_string();
    CHECK_LT(axis_index, num_axes()) << "axis " << axis_index
        << " out of range for " << num_axes() << "-D Tensor with shape "
        << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }

  /**
   * @brief Change the dimensions of the Tensor, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top Tensor during Layer::Reshape
   * or Layer::Forward. When changing the size of Tensor, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input Tensor and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  virtual void Reshape(const vector<int>& shape);

  virtual void Reshape(const TensorShape& shape);

  /// @brief Convenience method for Tensor whose dimensions are equal to or
  ///        less than 4.
  virtual void Reshape(const int dim0, const int dim1 = 1, const int dim2 = 1,
      const int dim3 = 1);

  virtual void ReshapeLike(const Tensor<Dtype>& that);

  virtual void resize(const size_t size);

  virtual bool ShapeEquals(const TensorProto& that);

  /**
   * @brief Copy from a source Tensor.
   *
   * @param source the Tensor to copy from
   * @param reshape if false, require this Tensor to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Tensor to other's
   *        shape if necessary
   */
  virtual void CopyFrom(const Tensor<Dtype>& source, bool reshape = false);

  virtual void FromProto(const TensorProto& proto, bool reshape = true);

  virtual void ToProto(TensorProto* proto);

  inline CBLAS_TRANSPOSE t() {
    if (transpose_ == CblasNoTrans) {
      transpose_ = CblasTrans;
    } else {
      transpose_ = CblasNoTrans;
    }
    return transpose_;
  }

  CBLAS_TRANSPOSE get_transpose() const {
    return transpose_;
  }

  virtual void malloc(const size_t num) {
    size_ = num;
    capacity_ = num;
    own_data_ = true;
  }

  virtual void free() {
    if (own_data_ && ptr_ != NULL) {
      ptr_ = NULL;
      size_ = 0;
      capacity_ = 0;
      own_data_ = false;
    }
  }

  virtual void choose_device() {
  }

  virtual Tensor<Dtype>& tensor() {
    return *this;
  }

  const void* ptr() const {
    return static_cast<const void*>(ptr_);
  }

  void* mutable_ptr() {
    return ptr_;
  }

  virtual const Dtype* data() const {
    return static_cast<const Dtype*>(ptr_);
  }

  virtual Dtype* mutable_data() {
    return static_cast<Dtype*>(ptr_);
  }

  inline Dtype at(const int n, const int c, const int h, const int w) {
    return data()[offset(n, c, h, w)];
  }

  inline Dtype at(const vector<int>& index) {
    return data()[offset(index)];
  }

  inline int offset(const int dim0, const int dim1 = 0, const int dim2 = 0,
      const int dim3 = 0) const {
    vector<int> indices(4);
    indices[0] = dim0;
    indices[1] = dim1;
    indices[2] = dim2;
    indices[3] = dim3;
    return offset(indices);
  }

  inline int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }

  inline int size() const {
    return size_;
  }

  inline int capacity() const {
    return capacity_;
  }

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  inline int size(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int size = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      size *= shape(i);
    }
    return size;
  }
  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  inline int size(int start_axis) const {
    return size(start_axis, num_axes());
  }

  inline int num_axes() const {
    return shape_.size();
  }

  inline const vector<int>& shape() const {
    return shape_;
  }

  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }

  inline static vector<int> shapeToVector(const TensorShape& shape) {
    CHECK_LE(shape.dim_size(), kMaxTensorAxes);
    vector<int> shape_vec(shape.dim_size());
    for (int i = 0; i < shape.dim_size(); ++i) {
      shape_vec[i] = shape.dim(i);
    }
    return shape_vec;
  }

  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << size_ << ")";
    return stream.str();
  }

// The following operations are inspired by Torch Math Functions
// The methods' @brief docs are from
// https://github.com/torch/torch7/blob/master/doc/maths.md
  /*
   * Construction or extraction functions
   */
  virtual void mem_set(const Dtype value) = 0;

//  static Dtype ones(Tensor* result, const vector<int> shape);
//
//  static Dtype ones(Tensor* result, const TensorShape& shape);
//
//  static Dtype ones(Tensor* result, const int dim0, int dim1 = 1,
//      int dim2 = 1, int dim3 = 1);
//
//  static Dtype zeros(Tensor* result, const vector<int> shape);
//
//  static Dtype zeros(Tensor* result, const TensorShape& shape);
//
//  static Dtype zeros(Tensor* result, const int dim0, int dim1 = 1,
//      int dim2 = 1, int dim3 = 1);

  virtual void zeros() {
    mem_set(0);
  }

  virtual void mem_copy(const Tensor<Dtype>& that) = 0;

  /*
   * Element-wise Mathematical Operations
   */
  virtual void abs(const Tensor<Dtype>& that) = 0;

  virtual void pow(const Tensor<Dtype>& that, const Dtype value) = 0;

  /**
   *  @brief Scale the Tensor data by a constant factor.
   */
  virtual void scale(Dtype scale_factor) = 0;

  /*
   * Basic operations
   */
  /**
   * @brief Add the given value to all elements in the that.
   */
  virtual void add(const Dtype value) = 0;

  /**
   *    @brief Add that to this that and put result into res.
   The number of elements must match, but sizes do not matter.
   */
  virtual void add(const Tensor<Dtype>& that) = 0;

  /**
   *    @brief Multiply elements of that by the scalar value and add it to this
   that. The number of elements must match, but sizes do not matter.
   */
  virtual void add(const Dtype value, const Tensor<Dtype>& that) = 0;

  /**
   *    @brief Performs the dot product between this that and that.
   The number of elements must match: both tensors are seen as a 1D
   vector.
   */
  virtual Dtype dot(const Tensor<Dtype>& that) = 0;

  /**
   * @brief Multiply all elements in the that by the given value.
   */
  virtual void mul(const Dtype value) = 0;

  /**
   * @brief Divide all elements in the that by the given value.
   */
  virtual void div(const Dtype value) = 0;

  /**
   *    @brief Element-wise multiplication of this that by that.
   The number of elements must match, but sizes do not matter.
   */
  virtual void cmul(const Tensor<Dtype>& first,
      const Tensor<Dtype>& second) = 0;

  /**
   *  @brief Performs the element-wise division of this that by that.
   The number of elements must match, but sizes do not matter.
   */
  virtual void cdiv(const Tensor<Dtype>& first,
      const Tensor<Dtype>& second) = 0;

  /**
   *    @brief Matrix vector product of this matrix (2D that) and vec.
   *    Sizes must respect the
   *    matrix-multiplication operation: if this is a n x m matrix, vec
   *    must be vector of size m and res must be a vector of size n.
   */
  virtual void mv(const Tensor<Dtype>& that) = 0;

  /**
   * @brief Performs a matrix-vector multiplication between mat (2D that) and
   *  vec2 (1D that) and add it to vec1.
   *  TODO:  Optional values v1 and v2 are scalars that multiply vec1 and vec2
   *  respectively. Optional value beta is a scalar that scales the result
   *  that, before accumulating the result into the that.
   *  Defaults to 1.0. In other words,
   *  res = beta * res + v1 * vec1 + v2 * mat * vec2
   *  Sizes must respect the matrix-multiplication operation:
   *  if mat is a n x m matrix, vec2 must be vector of size m and vec1
   *  must be a vector of size n.
   */
  virtual void addmv(const Tensor<Dtype>& that) = 0;

  /**
   * @brief Matrix matrix product of this matrix (2D that) and mat.
   *  If this is a n x m matrix, mat a m x p matrix, res must be a n x p
   *  matrix.
   */
  virtual void mm(const Tensor<Dtype>& that) = 0;

  /**
   * @brief Performs a matrix-matrix multiplication between this matrix
   *  (2D that) and that (2D that).
   *  Optional values v1 and v2 are scalars that multiply M and mat1 * mat2
   *  respectively.
   *  TODO: Optional value beta is a scalar that scales the result that,
   *  before accumulating the result into the that. Defaults to 1.0.
   *  In other words, res = res * beta + v1 * M + v2 * mat1*mat2
   *  If mat1 is a n x m matrix, mat2 a m x p matrix, M must be a n x p matrix.
   */
  virtual void addmm(const Tensor<Dtype>& that) = 0;

  /*
   * Global operations
   */
  /**
   * @brief Compute the sum of absolute values (L1 norm) of the data.
   */
  virtual Dtype asum() = 0;

  /**
   * @brief Compute the sum of squares (L2 norm squared) of the data.
   */
  virtual Dtype sumsq() = 0;

 protected:
  bool own_data_;
  TensorType type_;
  void* ptr_;
  vector<int> shape_;
  int size_;
  int capacity_;
  CBLAS_TRANSPOSE transpose_;

DISABLE_COPY_AND_ASSIGN(Tensor);
};
// class Tensor

template<typename Dtype>
class CPUTensor : public Tensor<Dtype> {
 public:
  CPUTensor();

  explicit CPUTensor(const vector<int>& shape);

  explicit CPUTensor(const int dim0, const int dim1 = 1, const int dim2 = 1,
      const int dim3 = 1);

  virtual ~CPUTensor();

//  virtual void Reshape(const vector<int>& shape);
//
//  virtual void Reshape(const TensorShape& shape);
//
//  virtual void Reshape(const int dim0, const int dim1 = 1, const int dim2 = 1,
//      const int dim3 = 1);
//
//  virtual void ReshapeLike(const Tensor<Dtype>& that);
//
//  virtual void resize(const size_t size);
//
//  virtual bool ShapeEquals(const TensorProto& that);
//
//  virtual void CopyFrom(const Tensor<Dtype>& source, bool reshape = false);
//
  virtual void FromProto(const TensorProto& proto, bool reshape = true);

  virtual void ToProto(TensorProto* proto);

  virtual void malloc(const size_t num);

  virtual void free();

  virtual void set_data(Dtype* data);

  virtual void mem_set(const Dtype value);

  virtual void mem_copy(const Tensor<Dtype>& that);

  virtual void abs(const Tensor<Dtype>& that);

  virtual void pow(const Tensor<Dtype>& that, const Dtype value);

  virtual void scale(Dtype scale_factor);

  virtual void add(const Dtype value);

  virtual void add(const Tensor<Dtype>& that);

  virtual void add(const Dtype value, const Tensor<Dtype>& that);

  virtual Dtype dot(const Tensor<Dtype>& that);

  virtual void mul(const Dtype value);

  virtual void div(const Dtype value);

  virtual void cmul(const Tensor<Dtype>& first, const Tensor<Dtype>& second);

  virtual void cdiv(const Tensor<Dtype>& first, const Tensor<Dtype>& second);

  virtual void mv(const Tensor<Dtype>& that);

  virtual void addmv(const Tensor<Dtype>& that);

  virtual void mm(const Tensor<Dtype>& that);

  virtual void addmm(const Tensor<Dtype>& that);

  virtual Dtype asum();

  virtual Dtype sumsq();

DISABLE_COPY_AND_ASSIGN(CPUTensor);
};
// class CPUTensor

#ifndef CPU_ONLY
template<typename Dtype>
class GPUTensor : public Tensor<Dtype> {
 public:
  GPUTensor();

  explicit GPUTensor(const vector<int>& shape);

  explicit GPUTensor(const int dim0, const int dim1 = 1, const int dim2 = 1,
      const int dim3 = 1);

  virtual ~GPUTensor();

//  virtual void Reshape(const vector<int>& shape);
//
//  virtual void Reshape(const TensorShape& shape);
//
//  virtual void Reshape(const int dim0, const int dim1 = 1, const int dim2 = 1,
//      const int dim3 = 1);
//
//  virtual void ReshapeLike(const Tensor<Dtype>& that);
//
//  virtual void resize(const size_t size);
//
//  virtual bool ShapeEquals(const TensorProto& that);
//
//  virtual void CopyFrom(const Tensor<Dtype>& source, bool reshape = false);
//
  virtual void FromProto(const TensorProto& proto, bool reshape = true);

  virtual void ToProto(TensorProto* proto);

  virtual void malloc(const size_t num);

  virtual void free();

  virtual void mem_set(const Dtype value);

  virtual void mem_copy(const Tensor<Dtype>& that);

  virtual void abs(const Tensor<Dtype>& that);

  virtual void pow(const Tensor<Dtype>& that, const Dtype value);

  virtual void scale(Dtype scale_factor);

  virtual void add(const Dtype value);

  virtual void add(const Tensor<Dtype>& that);

  virtual void add(const Dtype value, const Tensor<Dtype>& that);

  virtual Dtype dot(const Tensor<Dtype>& that);

  virtual void mul(const Dtype value);

  virtual void div(const Dtype value);

  virtual void cmul(const Tensor<Dtype>& first, const Tensor<Dtype>& second);

  virtual void cdiv(const Tensor<Dtype>& first, const Tensor<Dtype>& second);

  virtual void mv(const Tensor<Dtype>& that);

  virtual void addmv(const Tensor<Dtype>& that);

  virtual void mm(const Tensor<Dtype>& that);

  virtual void addmm(const Tensor<Dtype>& that);

  virtual Dtype asum();

  virtual Dtype sumsq();

DISABLE_COPY_AND_ASSIGN(GPUTensor);
};
// class GPUTensor
#endif  // #ifndef CPU_ONLY

template<typename Dtype>
class SyncedTensor : public Tensor<Dtype> {
 public:
  SyncedTensor();

  explicit SyncedTensor(const vector<int>& shape);

  explicit SyncedTensor(const int dim0, const int dim1 = 1, const int dim2 = 1,
      const int dim3 = 1);

  virtual ~SyncedTensor();

//  virtual void Reshape(const vector<int>& shape);
//
//  virtual void Reshape(const TensorShape& shape);
//
//  virtual void Reshape(const int dim0, const int dim1 = 1, const int dim2 = 1,
//      const int dim3 = 1);
//
//  virtual void ReshapeLike(const Tensor<Dtype>& that);
//
//  virtual void resize(const size_t size);
//
//  virtual bool ShapeEquals(const TensorProto& that);
//
//  virtual void CopyFrom(const Tensor<Dtype>& source, bool reshape = false);
//
  virtual void FromProto(const TensorProto& proto, bool reshape = true);

  virtual void ToProto(TensorProto* proto);

  virtual void malloc(const size_t num);

  virtual void free();

  virtual void mem_set(const Dtype value);

  virtual void mem_copy(const Tensor<Dtype>& that);

  virtual void choose_device();

  virtual Tensor<Dtype>& tensor() {
    choose_device();
    return *(current_tensor_.get());
  }

  void to_cpu();
  void to_gpu();
  const shared_ptr<Tensor<Dtype> > current_tensor() {
    choose_device();
    return current_tensor_;
  }
  const shared_ptr<CPUTensor<Dtype> > cpu_tensor() {
    to_cpu();
    return cpu_tensor_;
  }
#ifndef CPU_ONLY
  const shared_ptr<GPUTensor<Dtype> > gpu_tensor() {
    to_gpu();
    return gpu_tensor_;
  }
#endif  // #ifndef CPU_ONLY

  virtual const Dtype* data() {
    return tensor().data();
  }

  virtual Dtype* mutable_data() {
    return tensor().mutable_data();
  }
  const Dtype* cpu_data();
  Dtype* mutable_cpu_data();
  void set_cpu_data(Dtype* data);
  const Dtype* gpu_data();
  Dtype* mutable_gpu_data();

  enum SyncedHead {
    UNINITIALIZED,
    HEAD_AT_CPU,
    HEAD_AT_GPU,
    SYNCED
  };
  SyncedHead head() {
    return head_;
  }

  virtual void abs(const Tensor<Dtype>& that);

  virtual void pow(const Tensor<Dtype>& that, const Dtype value);

  virtual void scale(Dtype scale_factor);

  virtual void add(const Dtype value);

  virtual void add(const Tensor<Dtype>& that);

  virtual void add(const Dtype value, const Tensor<Dtype>& that);

  virtual Dtype dot(const Tensor<Dtype>& that);

  virtual void mul(const Dtype value);

  virtual void div(const Dtype value);

  virtual void cmul(const Tensor<Dtype>& first, const Tensor<Dtype>& second);

  virtual void cdiv(const Tensor<Dtype>& first, const Tensor<Dtype>& second);

  virtual void mv(const Tensor<Dtype>& that);

  virtual void addmv(const Tensor<Dtype>& that);

  virtual void mm(const Tensor<Dtype>& that);

  virtual void addmm(const Tensor<Dtype>& that);

  virtual Dtype asum();

  virtual Dtype sumsq();

 private:
  shared_ptr<CPUTensor<Dtype> > cpu_tensor_;
#ifndef CPU_ONLY
  shared_ptr<GPUTensor<Dtype> > gpu_tensor_;
#endif  // #ifndef CPU_ONLY
  shared_ptr<Tensor<Dtype> > current_tensor_;
  SyncedHead head_;
  bool own_cpu_data_;

DISABLE_COPY_AND_ASSIGN(SyncedTensor);
};
// class SyncedTensor

}  // namespace caffe

#endif  // CAFFE_TENSOR_HPP_
