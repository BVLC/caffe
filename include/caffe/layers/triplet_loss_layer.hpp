#ifndef CAFFE_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class TripletLossLayer : public LossLayer<Dtype> {
 public:
  explicit TripletLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_same_class_(), diff_diff_class_() {}

  void Reshape(const vector<Blob<Dtype>*>& bottom,
               const vector<Blob<Dtype>*>& top);

  inline const char* type() const { return "TripletLoss"; }
  inline int ExactNumBottomBlobs() const { return 3; }
  inline bool AllowForceBackward(const int bottom_index) const { return true; }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

 protected:
  /// @copydoc TripletLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_same_class_;
  Blob<Dtype> diff_diff_class_;
  Dtype alpha_;
  vector<Dtype> vec_loss_;
  int batch_size_;
  int vec_dimension_;
};

}  // namespace caffe

#endif  // CAFFE_TRIPLET_LOSS_LAYER_HPP_
