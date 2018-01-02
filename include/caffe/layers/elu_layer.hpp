#ifndef CAFFE_ELU_LAYER_HPP_
#define CAFFE_ELU_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Exponential Linear Unit non-linearity @f$
 *        Y = \left\{
 *        \begin{array}{lr}
 *            X                  & \mathrm{if} \; X > 0 \\
 *            \alpha (\exp(X)-1) & \mathrm{if} \; X \le 0
 *        \end{array} \right.
 *      @f$.  
 */
template<typename Dtype, typename MItype, typename MOtype>
class ELULayer : public NeuronLayer<Dtype, MItype, MOtype> {
 public:
  /**
   * @param param provides ELUParameter elu_param,
   *     with ELULayer options:
   *   - alpha (\b optional, default 1).
   *     the value @f$ \alpha @f$ by which controls saturation for
   *     negative inputs.
   */
  explicit ELULayer(const LayerParameter& param)
      : NeuronLayer<Dtype, MItype, MOtype>(param) {}
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  virtual inline const char* type() const { return "ELU"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (n \times c \times H \times W) @f$
   *      the inputs @f$ X @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (n \times c \times H \times W) @f$
   *      the computed outputs @f$
   *        Y = \left\{
   *        \begin{array}{lr}
   *            X                  & \mathrm{if} \; X > 0 \\
   *            \alpha (\exp(X)-1) & \mathrm{if} \; X \le 0
   *        \end{array} \right.
   *      @f$.  
   */
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the ELU inputs.
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
   *            1           & \mathrm{if} \; X > 0 \\
   *            Y + \alpha  & \mathrm{if} \; X \le 0
   *        \end{array} \right.
   *      @f$ if propagate_down[0].
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

#endif  // CAFFE_ELU_LAYER_HPP_
