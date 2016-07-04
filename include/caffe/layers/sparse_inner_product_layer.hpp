#ifndef CAFFE_SPARSE_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_SPARSE_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *        This layer also support sparse data (SparseBlob) as input
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template<typename Dtype>
class SparseInnerProductLayer : public InnerProductLayer<Dtype> {
 public:
  explicit SparseInnerProductLayer(const LayerParameter& param)
      : InnerProductLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "SparseInnerProduct"; }

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
};

}

#endif
