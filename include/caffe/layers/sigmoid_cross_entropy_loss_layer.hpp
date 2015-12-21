#ifndef CAFFE_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

/**
 * @brief Computes the cross-entropy (logistic) loss @f$
 *          E = \frac{-1}{n} \sum\limits_{n=1}^N \left[
 *                  p_n \log \hat{p}_n +
 *                  (1 - p_n) \log(1 - \hat{p}_n)
 *              \right]
 *        @f$, often used for predicting targets interpreted as probabilities.
 *
 * This layer is implemented rather than separate
 * SigmoidLayer + CrossEntropyLayer
 * as its gradient computation is more numerically stable.
 * At test time, this layer can be replaced simply by a SigmoidLayer.
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the scores @f$ x \in [-\infty, +\infty]@f$,
 *      which this layer maps to probability predictions
 *      @f$ \hat{p}_n = \sigma(x_n) \in [0, 1] @f$
 *      using the sigmoid function @f$ \sigma(.) @f$ (see SigmoidLayer).
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the targets @f$ y \in [0, 1] @f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed cross-entropy loss: @f$
 *          E = \frac{-1}{n} \sum\limits_{n=1}^N \left[
 *                  p_n \log \hat{p}_n + (1 - p_n) \log(1 - \hat{p}_n)
 *              \right]
 *      @f$
 */
template <typename Dtype>
class SigmoidCrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
  explicit SigmoidCrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
          sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
          sigmoid_output_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SigmoidCrossEntropyLoss"; }

 protected:
  /// @copydoc SigmoidCrossEntropyLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the sigmoid cross-entropy loss error gradient w.r.t. the
   *        predictions.
   *
   * Gradients cannot be computed with respect to the target inputs (bottom[1]),
   * so this method ignores bottom[1] and requires !propagate_down[1], crashing
   * if propagate_down[1] is set.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
   *      as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\ell_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
   *      (*Assuming that this top Blob is not used as a bottom (input) by any
   *      other layer of the Net.)
   * @param propagate_down see Layer::Backward.
   *      propagate_down[1] must be false as gradient computation with respect
   *      to the targets is not implemented.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$x@f$; Backward computes diff
   *      @f$ \frac{\partial E}{\partial x} =
   *          \frac{1}{n} \sum\limits_{n=1}^N (\hat{p}_n - p_n)
   *      @f$
   *   -# @f$ (N \times 1 \times 1 \times 1) @f$
   *      the labels -- ignored as we can't compute their error gradients
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// The internal SigmoidLayer used to map predictions to probabilities.
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  /// sigmoid_output stores the output of the SigmoidLayer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// top vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_top_vec_;
};

}  // namespace caffe

#endif  // CAFFE_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_
