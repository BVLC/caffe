#ifndef CAFFE_ATTENTION_LAYERS_HPP_
#define CAFFE_ATTENTION_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Equation (6) and 4.2.1 in
// [1] K. Xu, A. Courville, R. S. Zemel, and Y. Bengio,
// “Show, Attend and Tell : Neural Image Caption Generation with Visual Attention,” arXiv Prepr., 2015.
//
// bottom[0]: a        (N x C x H x W)
// bottom[1]: \alpha   (N x 1 x H x W)
// bottom[2]: \beta    (N x 1, or at least count == N)
// top[0]:    z        (N x C x 1 x 1)
//
//    z = \beta * per_channel_dot_product( a, \alpha )
template <typename Dtype>
class SoftAttentionLayer : public Layer<Dtype> {
 public:
  explicit SoftAttentionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {};
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SoftAttention"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_;
  int channels_;
  int spatial_dim_;
};

}  // namespace caffe

#endif  // CAFFE_ATTENTION_LAYERS_HPP_
