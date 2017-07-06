#ifndef CAFFE_LRN_LAYER_HPP_
#define CAFFE_LRN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/power_layer.hpp"
#include "caffe/layers/split_layer.hpp"

namespace caffe {

/**
 * @brief Normalize the input in a local region across or within feature maps.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class LRNLayer : public Layer<Dtype> {
 public:
  explicit LRNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LRN"; }
  virtual inline int_tp ExactNumBottomBlobs() const { return 1; }
  virtual inline int_tp ExactNumTopBlobs() const { return 1; }

  bool IsFused() const {
    return (this->layer_param_.lrn_param().fuse_type()
            != LRNParameter_FuseType_UNFUSED);
  }

  bool IsFusedWithPoolMax() const {
    return (this->layer_param_.lrn_param().fuse_type()
            == LRNParameter_FuseType_FUSED_POOL_MAX);
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void CrossChannelForward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void CrossChannelForward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void WithinChannelForward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void CrossChannelBackward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void CrossChannelBackward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void WithinChannelBackward(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void CrossChannelForward_fuse_pooling_gpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top,
      bool use_fuse);

  int_tp size_;
  int_tp pre_pad_;
  Dtype alpha_;
  Dtype beta_;
  Dtype k_;
  int_tp num_;
  int_tp channels_;
  int_tp height_;
  int_tp width_;

  int_tp pool_w_;
  int_tp pool_h_;
  int_tp pool_stride_w_;
  int_tp pool_stride_h_;
  int_tp pooled_width_;
  int_tp pooled_height_;
  bool fuse_tuned_;
  bool tuned_use_fuse_;
  Blob<Dtype> lrn_top_blob_;
  vector<Blob<Dtype>*> lrn_top_vec_;  // for pooling fusing

  // Fields used for normalization ACROSS_CHANNELS
  // scale_ stores the int_tpermediate summing results
  Blob<Dtype> scale_;

  // Fields used for normalization WITHIN_CHANNEL
  shared_ptr<SplitLayer<Dtype> > split_layer_;
  vector<Blob<Dtype>*> split_top_vec_;
  shared_ptr<PowerLayer<Dtype> > square_layer_;
  Blob<Dtype> square_input_;
  Blob<Dtype> square_output_;
  vector<Blob<Dtype>*> square_bottom_vec_;
  vector<Blob<Dtype>*> square_top_vec_;
  shared_ptr<PoolingLayer<Dtype> > pool_layer_;
  Blob<Dtype> pool_output_;
  vector<Blob<Dtype>*> pool_top_vec_;
  shared_ptr<PowerLayer<Dtype> > power_layer_;
  Blob<Dtype> power_output_;
  vector<Blob<Dtype>*> power_top_vec_;
  shared_ptr<EltwiseLayer<Dtype> > product_layer_;
  Blob<Dtype> product_input_;
  vector<Blob<Dtype>*> product_bottom_vec_;
};

}  // namespace caffe

#endif  // CAFFE_LRN_LAYER_HPP_
