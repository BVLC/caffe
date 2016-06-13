#ifndef CAFFE_EQUIV_POOLING_LAYER_HPP_
#define CAFFE_EQUIV_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Equivalent pooling layer can be used after dilated convolution layer
 * to provide equivalent mappings with original ConvNet. We call it dense net.
 * Dense net can be initialized by the baseline CNN's parameters, and can be
 * fine-tuning for a better performance than baseline CNN model.
 * For more details about equivalent pooling, please refer to: 
 * "Dense CNN Learning with Equivalent Mappings" by Jianxin Wu, Chen-Wei Xie and
 * Jian-Hao luo
 */
template <typename Dtype>
class EquivPoolingLayer : public Layer<Dtype> {
 public:
  explicit EquivPoolingLayer(const LayerParameter& param)
          : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "EquivPooling"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  int equiv_pooled_height_, equiv_pooled_width_;
  Blob<int> max_idx_;
};

}  // namespace caffe

#endif  // CAFFE_EQUIV_POOLING_LAYER_HPP_
