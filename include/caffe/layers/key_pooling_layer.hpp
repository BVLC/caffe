#ifndef CAFFE_KEY_POOLING_LAYER_HPP_
#define CAFFE_KEY_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/pooling_layer.hpp"

namespace caffe {

/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class KeyPoolingLayer : public Layer<Dtype> {
 public:
  explicit KeyPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param), pooling_layer_(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "KeyPooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  // The list of keys can also be an output
  virtual inline int MaxTopBlobs() const {
    return (this->layer_param_.pooling_param().pool() ==
            PoolingParameter_PoolMethod_MAX) ? 3 : 2;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // The following layer is used to perform the pooling per-key.
  PoolingLayer<Dtype> pooling_layer_;
  Blob<Dtype> key_top_mask_;
  // Store the keys for each of the elements in the generated top.
  vector<Dtype> has_keys_;
  // Store the start and end indices for each key in the input array.
  vector<int> key_start_;
  vector<int> key_len_;

};

}  // namespace caffe

#endif  // CAFFE_KEY_POOLING_LAYER_HPP_
