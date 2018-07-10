#ifndef CAFFE_INFOGAIN_LOSS_LAYER_HPP_
#define CAFFE_INFOGAIN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

/**
 * @brief A generalization of SoftmaxWithLossLayer that takes an
 *        "information gain" (infogain) matrix specifying the "value" of all label
 *        pairs.
 *
 * Equivalent to the SoftmaxWithLossLayer if the infogain matrix is the
 * identity.
 *
 * @param bottom input Blob vector (length 2-3)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ x @f$, a Blob with values in
 *      @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
 *      the @f$ K = CHW @f$ classes. This layer maps these scores to a
 *      probability distribution over classes using the softmax function
 *      @f$ \hat{p}_{nk} = \exp(x_{nk}) /
 *      \left[\sum_{k'} \exp(x_{nk'})\right] @f$ (see SoftmaxLayer).
 *   -# @f$ (N \times 1 \times 1 \times 1) @f$
 *      the labels @f$ l @f$, an integer-valued Blob with values
 *      @f$ l_n \in [0, 1, 2, ..., k - 1] @f$
 *      indicating the correct class label among the @f$ k @f$ classes
 *   -# @f$ (1 \times 1 \times k \times k) @f$
 *      (\b optional) the infogain matrix @f$ H @f$.  This must be provided as
 *      the third bottom blob input if not provided as the infogain_mat in the
 *      InfogainLossParameter. If @f$ H = I @f$, this layer is equivalent to the
 *      SoftmaxWithLossLayer.
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed infogain multinomial logistic loss: @f$ E =
 *        \frac{-1}{n} \sum\limits_{n=1}^n H_{l_n} \log(\hat{p}_n) =
 *        \frac{-1}{n} \sum\limits_{n=1}^n \sum\limits_{k=1}^{k} H_{l_n,k}
 *        \log(\hat{p}_{n,k})
 *      @f$, where @f$ H_{l_n} @f$ denotes row @f$l_n@f$ of @f$H@f$.
 */
template<typename Dtype, typename MItype, typename MOtype>
class InfogainLossLayer : public LossLayer<Dtype, MItype, MOtype> {
 public:
  explicit InfogainLossLayer(const LayerParameter& param)
      : LossLayer<Dtype, MItype, MOtype>(param), infogain_() {}
  virtual void LayerSetUp(
      const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Reshape(
      const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  // InfogainLossLayer takes 2-3 bottom Blobs; if there are 3 the third should
  // be the infogain matrix.  (Otherwise the infogain matrix is loaded from a
  // file specified by LayerParameter.)
  virtual inline int_tp ExactNumBottomBlobs() const { return -1; }
  virtual inline int_tp MinBottomBlobs() const { return 2; }
  virtual inline int_tp MaxBottomBlobs() const { return 3; }

  // InfogainLossLayer computes softmax prob internally.
  // optional second "top" outputs the softmax prob
  virtual inline int_tp ExactNumTopBlobs() const { return -1; }
  virtual inline int_tp MinTopBlobs() const { return 1; }
  virtual inline int_tp MaxTopBlobs() const { return 2; }

  virtual inline const char* type() const { return "InfogainLoss"; }

 protected:
  /// @copydoc InfogainLossLayer
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  /**
   * @brief Computes the infogain loss error gradient w.r.t. the predictions.
   *
   * Gradients cannot be computed with respect to the label inputs (bottom[1]),
   * so this method ignores bottom[1] and requires !propagate_down[1], crashing
   * if propagate_down[1] is set. (The same applies to the infogain matrix, if
   * provided as bottom[2] rather than in the layer_param.)
   *
   * @param top output Blob vector (length 1), providing the error gradient
   *      with respect to the outputs
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
   *      respect to the labels (similarly for propagate_down[2] and the
   *      infogain matrix, if provided as bottom[2])
   * @param bottom input Blob vector (length 2-3)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$ x @f$; Backward computes diff
   *      @f$ \frac{\partial E}{\partial x} @f$
   *   -# @f$ (N \times 1 \times 1 \times 1) @f$
   *      the labels -- ignored as we can't compute their error gradients
   *   -# @f$ (1 \times 1 \times k \times k) @f$
   *      (\b optional) the information gain matrix -- ignored as its error
   *      gradient computation is not implemented.
   */
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);

  /// Read the normalization mode parameter and compute the normalizer based
  /// on the blob size.  If normalization_mode is VALID, the count of valid
  /// outputs will be read from valid_count, unless it is -1 in which case
  /// all outputs are assumed to be valid.
  virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, int_tp valid_count);
  /// fill sum_rows_H_ according to matrix H
  virtual void sum_rows_of_H(const Blob<Dtype>* H);

  /// The internal SoftmaxLayer used to map predictions to a distribution.
  shared_ptr<Layer<Dtype, Dtype, Dtype> > softmax_layer_;
  /// prob stores the output probability predictions from the SoftmaxLayer.
  Blob<Dtype> prob_;
  /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  /// top vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_top_vec_;

  Blob<Dtype> infogain_;
  Blob<Dtype> sum_rows_H_;  // cache the row sums of H.

  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int_tp ignore_label_;
  /// How to normalize the output loss.
  LossParameter_NormalizationMode normalization_;

  int_tp infogain_axis_, outer_num_, inner_num_, num_labels_;
};

}  // namespace caffe

#endif  // CAFFE_INFOGAIN_LOSS_LAYER_HPP_
