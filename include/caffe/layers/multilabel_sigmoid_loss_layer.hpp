#ifndef CAFFE_MULTILABELSIGMOID_LOSS_LAYER_HPP_
#define CAFFE_MULTILABELSIGMOID_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

/* MultiLabelSigmoidLossLayer, now it only computes Sigmoid Loss,
but could be extended to use HingeLoss
*/
template <typename Dtype>
class MultiLabelSigmoidLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultiLabelSigmoidLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_(),
          sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
          sigmoid_output_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /*virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  */

  virtual inline const char* type() const { return "MultiLabelSigmoidLoss"; }

  virtual inline int MaxTopBlobs() const { return 2; }

  // We cannot backpropagate to the labels; ignore force_backward for these
  // inputs.
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }

 protected:
  /// @copydoc MultiLabelSigmoidLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;  // cached for backward pass

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

#endif  // CAFFE_MULTILABELSIGMOID_LOSS_LAYER_HPP_
