#ifndef CAFFE_POOLING_LAYER_HPP_
#define CAFFE_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 * For whole image processing, reducing redundancy.
 */
template<typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
  explicit PoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  virtual inline const char* type() const {
    return "Pooling";
  }
  virtual inline int_tp ExactNumBottomBlobs() const {
    return 1;
  }
  virtual inline int_tp MinTopBlobs() const {
    return 1;
  }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  virtual inline int_tp MaxTopBlobs() const {
    return
        (this->layer_param_.pooling_param().pool()
            == PoolingParameter_PoolMethod_MAX) ? 2 : 1;
  }

  Blob<int_tp> kernel_shape_;
  Blob<int_tp> ext_kernel_shape_;
  Blob<int_tp> stride_;
  Blob<int_tp> pad_;
  Blob<int_tp> dilation_;
  Blob<int_tp> size_;
  Blob<int_tp> pooled_size_;

  int_tp channel_axis_;
  int_tp num_spatial_axes_;
  int_tp channels_;

  bool use_skernel_;
  bool global_pooling_;

  int_tp max_top_blobs_;
  Blob<Dtype> rand_idx_;
  Blob<int_tp> max_idx_;
};

}  // namespace caffe

#endif  // CAFFE_POOLING_LAYER_HPP_
