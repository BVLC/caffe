#ifndef CAFFE_SWISH_LAYER_HPP_
#define CAFFE_SWISH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"


namespace caffe {

/**
 * @brief Swish non-linearity @f$ y = x \sigma (\beta x) @f$.
 *        A novel activation function that tends to work better than ReLU [1].
 *
 * [1] Prajit Ramachandran, Barret Zoph, Quoc V. Le. "Searching for
 *     Activation Functions". arXiv preprint arXiv:1710.05941v2 (2017).
 */
template<typename Dtype, typename MItype, typename MOtype>
class SwishLayer : public NeuronLayer<Dtype, MItype, MOtype> {
 public:
  /**
   * @param param provides SwishParameter swish_param,
   *     with SwishLayer options:
   *   - beta (\b optional, default 1).
   *     the value @f$ \beta @f$ in the @f$ y = x \sigma (\beta x) @f$.
   */
  explicit SwishLayer(const LayerParameter& param)
      : NeuronLayer<Dtype, MItype, MOtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Swish"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs @f$
   *        y = x \sigma (\beta x)
   *      @f$.
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
   *   -# @f$ (N \times C \times H \times W) @f$
   *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
   *      with respect to computed outputs @f$ y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial x}
   *            = \frac{\partial E}{\partial y}(\beta y +
   *              \sigma (\beta x)(1 - \beta y))
   *      @f$ if propagate_down[0]
   */
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<MItype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<MItype>*>& bottom);

  virtual void GenerateProgram();
};

}  // namespace caffe

#endif  // CAFFE_SWISH_LAYER_HPP_
