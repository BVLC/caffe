#ifndef CAFFE_POWER_LAYER_HPP_
#define CAFFE_POWER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Computes @f$ Y = (\alpha X + \beta) ^ \gamma @f$,
 *        as specified by the scale @f$ \alpha @f$, shift @f$ \beta @f$,
 *        and power @f$ \gamma @f$.
 */
template<typename Dtype, typename MItype, typename MOtype>
class PowerLayer : public NeuronLayer<Dtype, MItype, MOtype> {
 public:
  /**
   * @param param provides PowerParameter power_param,
   *     with PowerLayer options:
   *   - scale (\b optional, default 1) the scale @f$ \alpha @f$
   *   - shift (\b optional, default 0) the shift @f$ \beta @f$
   *   - power (\b optional, default 1) the power @f$ \gamma @f$
   */
  explicit PowerLayer(const LayerParameter& param)
      : NeuronLayer<Dtype, MItype, MOtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  virtual inline const char* type() const { return "Power"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (n \times c \times H \times W) @f$
   *      the inputs @f$ X @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (n \times c \times H \times W) @f$
   *      the computed outputs @f$
   *        Y = (\alpha X + \beta) ^ \gamma
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the power inputs.
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
   *        \frac{\partial E}{\partial X} =
   *            \frac{\partial E}{\partial Y}
   *            \alpha \gamma (\alpha X + \beta) ^ {\gamma - 1} =
   *            \frac{\partial E}{\partial Y}
   *            \frac{\alpha \gamma Y}{\alpha X + \beta}
   *      @f$ if propagate_down[0]
   */
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);

  /// @brief @f$ \gamma @f$ from layer_param_.power_param()
  Dtype power_;
  /// @brief @f$ \alpha @f$ from layer_param_.power_param()
  Dtype scale_;
  /// @brief @f$ \beta @f$ from layer_param_.power_param()
  Dtype shift_;
  /// @brief Result of @f$ \alpha \gamma @f$
  Dtype diff_scale_;
};

}  // namespace caffe

#endif  // CAFFE_POWER_LAYER_HPP_
