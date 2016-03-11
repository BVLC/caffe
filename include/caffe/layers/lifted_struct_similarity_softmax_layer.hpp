#ifndef CAFFE_LIFTED_STRUCT_SIMILARITY_LOSS_LAYER_HPP_
#define CAFFE_LIFTED_STRUCT_SIMILARITY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class LiftedStructSimilaritySoftmaxLossLayer : public LossLayer<Dtype> {
 public:
  explicit LiftedStructSimilaritySoftmaxLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline const char* type() const { return "LiftedStructSimilaritySoftmaxLoss"; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> dist_sq_;  // cached for backward pass
  Blob<Dtype> dot_;
  Blob<Dtype> ones_;
  Blob<Dtype> blob_pos_diff_;
  Blob<Dtype> blob_neg_diff_;
  Blob<Dtype> loss_aug_inference_;
  Blob<Dtype> summer_vec_;
  Dtype num_constraints;
};

}  // namespace caffe

#endif  // CAFFE_LIFTED_STRUCT_SIMILARITY_LOSS_LAYER_HPP_
