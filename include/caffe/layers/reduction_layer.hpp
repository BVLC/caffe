#ifndef CAFFE_REDUCTION_LAYER_HPP_
#define CAFFE_REDUCTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Compute "reductions" -- operations that return a scalar output Blob
 *        for an input Blob of arbitrary size, such as the sum, absolute sum,
 *        and sum of squares.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class ReductionLayer : public Layer<Dtype> {
 public:
  explicit ReductionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Reduction"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// @brief the reduction operation performed by the layer
  ReductionParameter_ReductionOp op_;
  /// @brief a scalar coefficient applied to all outputs
  Dtype coeff_;
  /// @brief the index of the first input axis to reduce
  int axis_;
  /// @brief the number of reductions performed
  int num_;
  /// @brief the input size of each reduction
  int dim_;
  /// @brief a helper Blob used for summation (op_ == SUM)
  Blob<Dtype> sum_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_REDUCTION_LAYER_HPP_
