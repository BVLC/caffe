#ifndef CAFFE_SCALE_LAYER_HPP_
#define CAFFE_SCALE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/bias_layer.hpp"

namespace caffe {

/**
 * @brief Computes the elementwise product of two input Blobs, with the shape of
 *        the latter Blob "broadcast" to match the shape of the former.
 *        Equivalent to tiling the latter Blob, then computing the elementwise
 *        product. Note: for efficiency and convenience, this layer can
 *        additionally perform a "broadcast" sum too when `bias_term: true`
 *        is set.
 *
 * The latter, scale input may be omitted, in which case it's learned as
 * parameter of the layer (as is the bias, if it is included).
 */
template<typename Dtype, typename MItype, typename MOtype>
class ScaleLayer: public Layer<Dtype, MItype, MOtype> {
 public:
  explicit ScaleLayer(const LayerParameter& param)
      : Layer<Dtype, MItype, MOtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  virtual inline const char* type() const { return "Scale"; }
  // Scale
  virtual inline int_tp MinBottomBlobs() const { return 1; }
  virtual inline int_tp MaxBottomBlobs() const { return 2; }
  virtual inline int_tp ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * In the below shape specifications, @f$ i @f$ denotes the value of the
   * `axis` field given by `this->layer_param_.scale_param().axis()`, after
   * canonicalization (i.e., conversion from negative to positive index,
   * if applicable).
   *
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (d_0 \times ... \times
   *           d_i \times ... \times d_j \times ... \times d_n) @f$
   *      the first factor @f$ X @f$
   *   -# @f$ (d_i \times ... \times d_j) @f$
   *      the second factor @f$ Y @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (d_0 \times ... \times
   *           d_i \times ... \times d_j \times ... \times d_n) @f$
   *      the product @f$ z = X Y @f$ computed after "broadcasting" Y.
   *      Equivalent to tiling @f$ Y @f$ to have the same shape as @f$ X @f$,
   *      then computing the elementwise product.
   */
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);

  void GenerateProgram();

  shared_ptr<Layer<Dtype, Dtype, Dtype> > bias_layer_;
  vector<Blob<Dtype>*> bias_bottom_vec_;
  vector<bool> bias_propagate_down_;
  int_tp bias_param_id_;

  Blob<Dtype> sum_multiplier_;
  Blob<Dtype> sum_result_;
  Blob<Dtype> temp_;
  int_tp axis_;
  int_tp outer_dim_, scale_dim_, inner_dim_;
};


}  // namespace caffe

#endif  // CAFFE_SCALE_LAYER_HPP_
