#ifndef CAFFE_SIGNED_SQRT_LAYER_HPP_
#define CAFFE_SIGNED_SQRT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class SignedSqrtLayer : public NeuronLayer<Dtype> {
 public:
  explicit SignedSqrtLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SignedSqrt"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // the epsilon that is added to the denominator, to prevent 1.0/0.0
  static const Dtype epsilon = 0.03;
};

}  // namespace caffe

#endif  // CAFFE_SIGNED_SQRT_LAYER_HPP_
