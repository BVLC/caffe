#ifndef CAFFE_ARGMAX_LAYER_HPP_
#define CAFFE_ARGMAX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Compute the index of the @f$ k @f$ max values for each datum across
 *        all dimensions @f$ (c \times H \times W) @f$.
 *
 * intended for use after a classification layer to produce a prediction.
 * If parameter out_max_val is set to true, output is a vector of pairs
 * (max_ind, max_val) for each image. The axis parameter specifies an axis
 * along which to maximise.
 *
 * NOTE: does not implement Backwards operation.
 */
template<typename Dtype, typename MItype, typename MOtype>
class ArgMaxLayer : public Layer<Dtype, MItype, MOtype> {
 public:
  /**
   * @param param provides ArgMaxParameter argmax_param,
   *     with ArgMaxLayer options:
   *   - top_k (\b optional uint_tp_tp, default 1).
   *     the number @f$ k @f$ of maximal items to output.
   *   - out_max_val (\b optional bool, default false).
   *     if set, output a vector of pairs (max_ind, max_val)
   *     unless axis is set then
   *     output max_val along the specified axis.
   *   - axis (\b optional int_tp).
   *     if set, maximise along the specified axis else maximise the flattened
   *     trailing dimensions for each index of the first / num dimension.
   */
  explicit ArgMaxLayer(const LayerParameter& param)
      : Layer<Dtype, MItype, MOtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  virtual inline const char* type() const { return "ArgMax"; }
  virtual inline int_tp ExactNumBottomBlobs() const { return 1; }
  virtual inline int_tp ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (n \times c \times H \times W) @f$
   *      the inputs @f$ X @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (n \times 1 \times k) @f$ or, if out_max_val
   *      @f$ (n \times 2 \times k) @f$ unless axis set than e.g.
   *      @f$ (n \times k \times H \times W) @f$ if axis == 1
   *      the computed outputs @f$
   *       y_n = \arg\max\limits_i x_{ni}
   *      @f$ (for @f$ k = 1 @f$).
   */
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  bool out_max_val_;
  size_t top_k_;
  bool has_axis_;
  int_tp axis_;
};

}  // namespace caffe

#endif  // CAFFE_ARGMAX_LAYER_HPP_
