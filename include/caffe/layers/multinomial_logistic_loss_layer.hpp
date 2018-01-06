#ifndef CAFFE_MULTINOMIAL_LOGISTIC_LOSS_LAYER_HPP_
#define CAFFE_MULTINOMIAL_LOGISTIC_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the multinomial logistic loss for a one-of-many
 *        classification task, directly taking a predicted probability
 *        distribution as input.
 *
 * When predictions are not already a probability distribution, you should
 * instead use the SoftmaxWithLossLayer, which maps predictions to a
 * distribution using the SoftmaxLayer, before computing the multinomial
 * logistic loss. The SoftmaxWithLossLayer should be preferred over separate
 * SoftmaxLayer + MultinomialLogisticLossLayer
 * as its gradient computation is more numerically stable.
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (n \times c \times H \times W) @f$
 *      the predictions @f$ \hat{p} @f$, a Blob with values in
 *      @f$ [0, 1] @f$ indicating the predicted probability of each of the
 *      @f$ k = CHW @f$ classes.  Each prediction vector @f$ \hat{p}_n @f$
 *      should sum to 1 as in a probability distribution: @f$
 *      \forall n \sum\limits_{k=1}^k \hat{p}_{nk} = 1 @f$.
 *   -# @f$ (n \times 1 \times 1 \times 1) @f$
 *      the labels @f$ l @f$, an integer-valued Blob with values
 *      @f$ l_n \in [0, 1, 2, ..., k - 1] @f$
 *      indicating the correct class label among the @f$ k @f$ classes
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed multinomial logistic loss: @f$ E =
 *        \frac{-1}{n} \sum\limits_{n=1}^n \log(\hat{p}_{n,l_n})
 *      @f$
 */
template<typename Dtype, typename MItype, typename MOtype>
class MultinomialLogisticLossLayer
    : public LossLayer<Dtype, MItype, MOtype> {
 public:
  explicit MultinomialLogisticLossLayer(const LayerParameter& param)
      : LossLayer<Dtype, MItype, MOtype>(param) {}
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  virtual inline const char* type() const { return "MultinomialLogisticLoss"; }

 protected:
  /// @copydoc MultinomialLogisticLossLayer
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  /**
   * @brief Computes the multinomial logistic loss error gradient w.r.t. the
   *        predictions.
   *
   * Gradients cannot be computed with respect to the label inputs (bottom[1]),
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
   *      propagate_down[1] must be false as we can't compute gradients with
   *      respect to the labels.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (n \times c \times H \times W) @f$
   *      the predictions @f$ \hat{p} @f$; Backward computes diff
   *      @f$ \frac{\partial E}{\partial \hat{p}} @f$
   *   -# @f$ (n \times 1 \times 1 \times 1) @f$
   *      the labels -- ignored as we can't compute their error gradients
   */
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_MULTINOMIAL_LOGISTIC_LOSS_LAYER_HPP_
