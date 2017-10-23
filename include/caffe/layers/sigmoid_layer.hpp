#ifndef CAFFE_SIGMOID_LAYER_HPP_
#define CAFFE_SIGMOID_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Sigmoid function non-linearity @f$
 *         Y = (1 + \exp(-X))^{-1}
 *     @f$, a classic choice in neural networks.
 *
 * Note that the gradient vanishes as the values move away from 0.
 * The ReLULayer is often a better choice for this reason.
 */
template<typename Dtype, typename MItype, typename MOtype>
class SigmoidLayer : public NeuronLayer<Dtype, MItype, MOtype> {
 public:
  explicit SigmoidLayer(const LayerParameter& param)
      : NeuronLayer<Dtype, MItype, MOtype>(param) {}
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual inline const char* type() const { return "Sigmoid"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (n \times c \times H \times W) @f$
   *      the inputs @f$ X @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (n \times c \times H \times W) @f$
   *      the computed outputs @f$
   *        Y = (1 + \exp(-X))^{-1}
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the sigmoid inputs.
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
   *        \frac{\partial E}{\partial X}
   *            = \frac{\partial E}{\partial Y} Y (1 - Y)
   *      @f$ if propagate_down[0]
   */
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<MItype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<MItype>*>& bottom);

  virtual void GenerateProgram();
};

}  // namespace caffe

#endif  // CAFFE_SIGMOID_LAYER_HPP_
