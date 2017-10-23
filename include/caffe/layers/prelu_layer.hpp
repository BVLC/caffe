#ifndef CAFFE_PRELU_LAYER_HPP_
#define CAFFE_PRELU_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Parameterized Rectified Linear Unit non-linearity @f$
 *        y_i = \max(0, x_i) + a_i \min(0, x_i)
 *        @f$. The differences from ReLULayer are 1) negative slopes are
 *        learnable though backprop and 2) negative slopes can vary across
 *        channels. The number of axes of input blob should be greater than or
 *        equal to 2. The 1st axis (0-based) is seen as channels.
 */
template<typename Dtype, typename MItype, typename MOtype>
class PReLULayer : public NeuronLayer<Dtype, MItype, MOtype> {
 public:
  /**
   * @param param provides PReLUParameter prelu_param,
   *     with PReLULayer options:
   *   - filler (\b optional, FillerParameter,
   *     default {'type': constant 'value':0.25}).
   *   - channel_shared (\b optional, default false).
   *     negative slopes are shared across channels.
   */
  explicit PReLULayer(const LayerParameter& param)
      : NeuronLayer<Dtype, MItype, MOtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  virtual inline const char* type() const { return "PReLU"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (n \times c \times ...) @f$
   *      the inputs @f$ X @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (n \times c \times ...) @f$
   *      the computed outputs for each channel @f$i@f$ @f$
   *        y_i = \max(0, x_i) + a_i \min(0, x_i)
   *      @f$.
   */
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the PReLU inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (n \times c \times ...) @f$
   *      containing error gradients @f$ \frac{\partial E}{\partial Y} @f$
   *      with respect to computed outputs @f$ Y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (n \times c \times ...) @f$
   *      the inputs @f$ X @f$; For each channel @f$i@f$, backward fills their
   *      diff with gradients @f$
   *        \frac{\partial E}{\partial x_i} = \left\{
   *        \begin{array}{lr}
   *            a_i \frac{\partial E}{\partial y_i} & \mathrm{if} \; x_i \le 0 \\
   *            \frac{\partial E}{\partial y_i} & \mathrm{if} \; x_i > 0
   *        \end{array} \right.
   *      @f$.
   *      If param_propagate_down_[0] is true, it fills the diff with gradients
   *      @f$
   *        \frac{\partial E}{\partial a_i} = \left\{
   *        \begin{array}{lr}
   *            \sum_{x_i} x_i \frac{\partial E}{\partial y_i} & \mathrm{if} \; x_i \le 0 \\
   *            0 & \mathrm{if} \; x_i > 0
   *        \end{array} \right.
   *      @f$.
   */
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);

  virtual void GenerateProgram();

  bool channel_shared_;
  // dot multiplier for backward computation of params
  Blob<Dtype> multiplier_;
  // temporary buffer for backward computation
  Blob<Dtype> backward_buff_;
  // memory for in-place computation
  Blob<Dtype> bottom_memory_;
};

}  // namespace caffe

#endif  // CAFFE_PRELU_LAYER_HPP_
