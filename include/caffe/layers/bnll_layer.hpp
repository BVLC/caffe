#ifndef CAFFE_BNLL_LAYER_HPP_
#define CAFFE_BNLL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Computes @f$ Y = X + \log(1 + \exp(-X)) @f$ if @f$ X > 0 @f$;
 *        @f$ Y = \log(1 + \exp(X)) @f$ otherwise.
 *
 * @param bottom input Blob vector (length 1)
 *   -# @f$ (n \times c \times H \times W) @f$
 *      the inputs @f$ X @f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (n \times c \times H \times W) @f$
 *      the computed outputs @f$
 *      Y = \left\{
 *         \begin{array}{ll}
 *            X + \log(1 + \exp(-X)) & \mbox{if } X > 0 \\
 *            \log(1 + \exp(X)) & \mbox{otherwise}
 *         \end{array} \right.
 *      @f$
 */
template<typename Dtype, typename MItype, typename MOtype>
class BNLLLayer : public NeuronLayer<Dtype, MItype, MOtype> {
 public:
  explicit BNLLLayer(const LayerParameter& param)
      : NeuronLayer<Dtype, MItype, MOtype>(param) {}

  virtual inline const char* type() const { return "BNLL"; }

 protected:
  /// @copydoc BNLLLayer
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the BNLL inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (n \times c \times H \times W) @f$
   *      containing error gradients @f$ \frac{\partial E}{\partial Y} @f$
   *      with respect to computed outputs @f$ Y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (n \times c \times H \times W) @f$
   *      the inputs @f$ X @f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial X}
   *      @f$ if propagate_down[0]
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

#endif  // CAFFE_BNLL_LAYER_HPP_
