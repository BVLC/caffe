#ifndef CAFFE_COSINE_LOSS_LAYER_HPP_
#define CAFFE_COSINE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the Cosine loss @f$
 *          E = \frac{1}{N} \sum\limits_{n=1}^N 1 - \frac{\hat{y}_n^T y_n}
 *        {\left| \left| \hat{y}_n \right| \right|_2 
 *         \left| \left| y_n \right| \right|_2} @f$
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ \hat{y} \in [-\infty, +\infty]@f$
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the targets @f$ y \in [-\infty, +\infty]@f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed Cosine loss: @f$ E = 
 *        \frac{1}{N} \sum\limits_{n=1}^N 1 - \frac{\hat{y}_n^T y_n}
 *        {\left| \left| \hat{y}_n \right| \right|_2 
 *         \left| \left| y_n \right| \right|_2} @f$
 */
template <typename Dtype>
class CosineLossLayer : public LossLayer<Dtype> {
 public:
   /**
    * @param param provides LossParameter loss_param, with normalization
    *        options similar to the SoftmaxWithLoss and 
    *        SigmoidCrossEntropyLoss layers.
    *        The ignore_label option is not supported.
    */
  explicit CosineLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CosineLoss"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the Cosine error gradient w.r.t. the inputs.
   * 
   * Similar to the EuclideanLossLayer, this layer can propagate back gradients
   * to the label inputs bottom[1] (but still only will if propagate_down[1] is
   * set, due to being produced by learnable parameters or if force_backward is
   * set). Similarly, it is "commutative" as well.
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
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$\hat{y}@f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial \hat{y}} =
   *            \frac{1}{\left| \left| y \right| \right|_2 \cdot
   *                     \left| \left| \hat{y} \right| \right|_2} 
   *            \left( \frac{\hat{y}^T y}
   *                        {\left| \left| \hat{y} \right| \right|_2^2} \hat{y}
   *                   - y
   *            \right)
   *      @f$ if propagate_down[0]
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the targets @f$y@f$; Backward fills their diff with gradients
   *      @f$ \frac{\partial E}{\partial y} =
   *            \frac{1}{\left| \left| y \right| \right|_2 \cdot
   *                     \left| \left| \hat{y} \right| \right|_2} 
   *            \left( \frac{y^T \hat{y}}
   *                        {\left| \left| y \right| \right|_2^2} y
   *                   - \hat{y}
   *            \right)
   *      @f$ if propagate_down[1]
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// Read the normalization mode parameter and compute the normalizer based
  /// on the blob size.
  virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, int valid_count);

  int outer_num_;
  int inner_num_;
  int cosine_axis_;
  /// Caches the dot products.
  Blob<Dtype> dots_;
  /// Caches the lengths of the input vectors.
  Blob<Dtype> len_inp_;
  /// Caches the lengths of the target vectors.
  Blob<Dtype> len_label_;
  /// How to normalize the output loss.
  LossParameter_NormalizationMode normalization_;
};

}  // namespace caffe

#endif  // CAFFE_COSINE_LOSS_LAYER_HPP_
