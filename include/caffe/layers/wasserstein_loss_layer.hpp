#ifndef CAFFE_WASSERSTEIN_LOSS_LAYER_HPP_
#define CAFFE_WASSERSTEIN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Wasserstein loss layer.
 *
 * Computes min_{T s.t. T1 = h(x), T^t1 = y} <T, M> - (1/lambda) H(T)
 * with M a ground metric matrix and H(T) = -<T, log T>
 * 
 * Expects normalized inputs, so that the sum of predicted values equals
 * the sum of label values, for each sample.
 *
 * Requires a ground metric matrix of dimension d x d, where d is the number of 
 * label dimensions. Specify via the ground_metric layer parameter.
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      Predictions @f$ \hat{p} @f$, a Blob with values in
 *      @f$ [0, 1] @f$ indicating the predicted probability of each of the
 *      @f$ K = CHW @f$ classes.  Each prediction vector @f$ \hat{p}_n @f$
 *      should sum to 1 as in a probability distribution: @f$
 *      \forall n \sum\limits_{k=1}^K \hat{p}_{nk} = 1 @f$.
 *   -# @f$ (N \times 1 \times 1 \times 1) @f$
 *      Labels @f$ l @f$, an integer-valued Blob with values
 *      @f$ l_n \in [0, 1, 2, ..., K - 1] @f$
 *      indicating the correct class label among the @f$ K @f$ classes,
 *      OR @f$ (N \times C \times H \times W) @f$
 *      Labels @f$ l @f$, with values in @f$ [0,1] @f$, normalized to
 *      sum to 1: @f$ \forall n \sum\limits_{k=1}^K l_{nk} = 1 @f$.
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      The computed loss: @f$ E = 
 *      \frac{1}{n} min_{T s.t. T1 = \hat{p}, T^t1 = y} <T, M> - (1/lambda) H(T) @f$
 */
template <typename Dtype>
class WassersteinLossLayer : public LossLayer<Dtype> {
 public:
  explicit WassersteinLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "WassersteinLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /// @copydoc WassersteinLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes Wasserstein error gradient with respect to the
   * predictions.
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
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$ \hat{p} @f$; Backward computes diff
   *      @f$ \frac{\partial E}{\partial \hat{p}} @f$
   *   -# @f$ (N \times 1 \times 1 \times 1) @f$
   *      the labels -- ignored as we can't compute their error gradients
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> groundm_; // ground metric
  Blob<Dtype> u_; // row scaling element
  Blob<Dtype> v_; // column scaling element
  Blob<Dtype> K_; // e^(-lambda*groundm)
  Blob<Dtype> KM_; // K * groundm
  Blob<Dtype> KlogK_; // K * log(K)
  Blob<Dtype> one_; // constant matrix
  Blob<Dtype> tmp_; // temp
};






}  // namespace caffe

#endif  // CAFFE_WASSERSTEIN_LOSS_LAYER_HPP_
