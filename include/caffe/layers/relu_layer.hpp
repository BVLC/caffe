#ifndef CAFFE_RELU_LAYER_HPP_
#define CAFFE_RELU_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Rectified Linear Unit non-linearity @f$ Y = \max(0, X) @f$.
 *        The simple max is fast to compute, and the function does not saturate.
 */
template<typename Dtype, typename MItype, typename MOtype>
class ReLULayer : public NeuronLayer<Dtype, MItype, MOtype> {
 public:
  /**
   * @param param provides ReLUParameter relu_param,
   *     with ReLULayer options:
   *   - negative_slope (\b optional, default 0).
   *     the value @f$ \nu @f$ by which negative values are multiplied.
   */
  explicit ReLULayer(const LayerParameter& param)
      : NeuronLayer<Dtype, MItype, MOtype>(param) {}
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual inline const char* type() const { return "ReLU"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (n \times c \times H \times W) @f$
   *      the inputs @f$ X @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (n \times c \times H \times W) @f$
   *      the computed outputs @f$
   *        Y = \max(0, X)
   *      @f$ by default.  If a non-zero negative_slope @f$ \nu @f$ is provided,
   *      the computed outputs are @f$ Y = \max(0, X) + \nu \min(0, X) @f$.
   */
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the ReLU inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (n \times c \times H \times W) @f$
   *      containing error gradients @f$ \frac{\partial E}{\partial Y} @f$
   *      with respect to computed outputs @f$ Y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (n \times c \times H \times W) @f$
   *      the inputs @f$ X @f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial X} = \left\{
   *        \begin{array}{lr}
   *            0 & \mathrm{if} \; X \le 0 \\
   *            \frac{\partial E}{\partial Y} & \mathrm{if} \; X > 0
   *        \end{array} \right.
   *      @f$ if propagate_down[0], by default.
   *      If a non-zero negative_slope @f$ \nu @f$ is provided,
   *      the computed gradients are @f$
   *        \frac{\partial E}{\partial X} = \left\{
   *        \begin{array}{lr}
   *            \nu \frac{\partial E}{\partial Y} & \mathrm{if} \; X \le 0 \\
   *            \frac{\partial E}{\partial Y} & \mathrm{if} \; X > 0
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
};

}  // namespace caffe

#endif  // CAFFE_RELU_LAYER_HPP_
