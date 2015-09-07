#ifndef CAFFE_TENSOR_HPP_
#define CAFFE_TENSOR_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

const int kMaxTensorAxes = INT_MAX;

namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 */
template <typename Dtype>
class Tensor {
 public:
  Tensor()
       : mem_(), count_(0), capacity_(0) {}

  explicit Tensor(const vector<int>& shape);

  void Reshape(const vector<int>& shape);
  void ReshapeLike(const Tensor& other);
  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }
  inline const vector<int>& shape() const { return shape_; }
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   */
  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }
  inline int num_axes() const { return shape_.size(); }
  inline int count() const { return count_; }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   */
  inline int CanonicalAxisIndex(int axis_index) const {
    ASSERT(axis_index >= -num_axes(),
        "axis " << axis_index << " out of range for " << num_axes()
        << "-D Tensor with shape " << shape_string());
    ASSERT(axis_index < num_axes(),
        "axis " << axis_index << " out of range for " << num_axes()
        << "-D Tensor with shape " << shape_string());
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }

  inline int offset(const vector<int>& indices) const {
    ASSERT(indices.size() <= num_axes(), "");
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        ASSERT(indices[i] >= 0, "");
        ASSERT(indices[i] < shape(i), "");
        offset += indices[i];
      }
    }
    return offset;
  }

  void CopyFrom(const Tensor<Dtype>& source);
  void CopyChunkFrom(const Tensor<Dtype>& source, int count,
    int this_offset, int other_offset);

  inline Dtype mem_at(const vector<int>& index) const {
    return cpu_mem()[offset(index)];
  }

  inline const shared_ptr<SyncedMemory>& mem() const {
    ASSERT(mem_, "");
    return mem_;
  }

  const Dtype* cpu_mem() const;
  void set_cpu_mem(Dtype* data);
  const Dtype* gpu_mem() const;
  Dtype* mutable_cpu_mem();
  Dtype* mutable_gpu_mem();

  /// @brief Compute the sum of absolute values (L1 norm) of the mem.
  Dtype asum() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the mem.
  Dtype sumsq() const;

  /// @brief Scale the mem by a constant factor.
  void scale(Dtype scale_factor);

  void ShareMem(const Tensor& other);

  bool Initialized() { return mem_ ? true : false; }
  void SetValues(const Dtype value);
  void MulFrom(const Tensor& source);
  void AddFrom(const Tensor& source);
  // NOLINT_NEXT_LINE(runtime/int)
  void AddFromGPUPointer(Dtype* ptr, long long size);
  void AddMulFrom(const Tensor& source, Dtype alpha);
  void AddMulFromDynamicMode(const Tensor& source, Dtype alpha);
  Dtype DotPFrom(const Tensor& source);
  Caffe::Brew mode() {
    return Caffe::mode();
  }

 protected:
  shared_ptr<SyncedMemory> mem_;
  vector<int> shape_;
  int count_;
  int capacity_;

  DISABLE_COPY_AND_ASSIGN(Tensor);
};  // class Tensor

}  // namespace caffe

#endif  // CAFFE_TENSOR_HPP_
