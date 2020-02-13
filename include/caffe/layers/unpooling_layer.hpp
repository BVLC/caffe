#ifndef CAFFE_UNPOOLING_LAYER_HPP_
#define CAFFE_UNPOOLING_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/pooling_layer.hpp"

namespace caffe {

/**
 * @brief Unpools the input image by undoing the max, average, etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class UnpoolingLayer : public Layer<Dtype> {
 public:
  explicit UnpoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Unpooling"; }
  virtual inline int MinBottomBlobs() const {
    return (this->layer_param_.pooling_param().pool() ==
            PoolingParameter_PoolMethod_MAX) ? 2 : 1;
  }
  virtual inline int MaxBottomBlobs() const {
    return (this->layer_param_.pooling_param().pool() ==
            PoolingParameter_PoolMethod_MAX) ? 3 : 2;
  }
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

  int kernel_size_;
  int stride_;
  int pad_;
  int channels_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  bool global_pooling_;
  Blob<Dtype> pool_count_;
  Blob<int> max_idx_;
};

}  // namespace caffe

#endif  // CAFFE_UNPOOLING_LAYER_HPP_
