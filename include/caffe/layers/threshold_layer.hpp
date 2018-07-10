#ifndef CAFFE_THRESHOLD_LAYER_HPP_
#define CAFFE_THRESHOLD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Tests whether the input exceeds a threshold: outputs 1 for inputs
 *        above threshold; 0 otherwise.
 */
template<typename Dtype, typename MItype, typename MOtype>
class ThresholdLayer : public NeuronLayer<Dtype, MItype, MOtype> {
 public:
  /**
   * @param param provides ThresholdParameter threshold_param,
   *     with ThresholdLayer options:
   *   - threshold (\b optional, default 0).
   *     the threshold value @f$ t @f$ to which the input values are compared.
   */
  explicit ThresholdLayer(const LayerParameter& param)
      : NeuronLayer<Dtype, MItype, MOtype>(param) {}
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
                       const vector<Blob<MOtype>*>& top);
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
                          const vector<Blob<MOtype>*>& top);

  virtual inline const char* type() const { return "Threshold"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (n \times c \times H \times W) @f$
   *      the inputs @f$ X @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (n \times c \times H \times W) @f$
   *      the computed outputs @f$
   *       Y = \left\{
   *       \begin{array}{lr}
   *         0 & \mathrm{if} \; X \le t \\
   *         1 & \mathrm{if} \; X > t
   *       \end{array} \right.
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  virtual void GenerateProgram();

  Dtype threshold_;
};

}  // namespace caffe

#endif  // CAFFE_THRESHOLD_LAYER_HPP_
