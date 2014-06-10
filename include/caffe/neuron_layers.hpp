// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_NEURON_LAYERS_HPP_
#define CAFFE_NEURON_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "leveldb/db.h"
#include "pthread.h"
#include "boost/scoped_ptr.hpp"
#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#define HDF5_DATA_DATASET_NAME "data"
#define HDF5_DATA_LABEL_NAME "label"

namespace caffe {

/* NeuronLayer
  An interface for layers that take one blob as input (x),
  and produce one blob as output (y).
*/
template <typename Dtype>
class NeuronLayer : public Layer<Dtype> {
 public:
  explicit NeuronLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_NONE;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
};

/* BNLLLayer

  y = x + log(1 + exp(-x))  if x > 0
  y = log(1 + exp(x))       if x <= 0

  y' = exp(x) / (exp(x) + 1)
*/
template <typename Dtype>
class BNLLLayer : public NeuronLayer<Dtype> {
 public:
  explicit BNLLLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_BNLL;
  }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
};

/* DropoutLayer
  During training only, sets some portion of x to 0, adjusting the
  vector magnitude accordingly.

  mask = bernoulli(1 - threshold)
  scale = 1 / (1 - threshold)
  y = x * mask * scale

  y' = mask * scale
*/
template <typename Dtype>
class DropoutLayer : public NeuronLayer<Dtype> {
 public:
  explicit DropoutLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_DROPOUT;
  }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  shared_ptr<Blob<unsigned int> > rand_vec_;
  Dtype threshold_;
  Dtype scale_;
  unsigned int uint_thres_;
};

/* PowerLayer
  y = (shift + scale * x) ^ power

  y' = scale * power * (shift + scale * x) ^ (power - 1)
     = scale * power * y / (shift + scale * x)
*/
template <typename Dtype>
class PowerLayer : public NeuronLayer<Dtype> {
 public:
  explicit PowerLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_POWER;
  }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  Dtype power_;
  Dtype scale_;
  Dtype shift_;
  Dtype diff_scale_;
};

/* ReLULayer
  Rectified Linear Unit non-linearity.
  The simple max is fast to compute, and the function does not saturate.

  y = max(0, x).

  y' = 0  if x < 0
  y' = 1 if x > 0
*/
template <typename Dtype>
class ReLULayer : public NeuronLayer<Dtype> {
 public:
  explicit ReLULayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_RELU;
  }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
};

/* SigmoidLayer
  Sigmoid function non-linearity, a classic choice in neural networks.
  Note that the gradient vanishes as the values move away from 0.
  The ReLULayer is often a better choice for this reason.

  y = 1. / (1 + exp(-x))

  y ' = exp(x) / (1 + exp(x))^2
  or
  y' = y * (1 - y)
*/
template <typename Dtype>
class SigmoidLayer : public NeuronLayer<Dtype> {
 public:
  explicit SigmoidLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_SIGMOID;
  }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
};

/* TanHLayer
  Hyperbolic tangent non-linearity, popular in auto-encoders.

  y = 1. * (exp(2x) - 1) / (exp(2x) + 1)

  y' = 1 - ( (exp(2x) - 1) / (exp(2x) + 1) ) ^ 2
*/
template <typename Dtype>
class TanHLayer : public NeuronLayer<Dtype> {
 public:
  explicit TanHLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_TANH;
  }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
};

/* ThresholdLayer
  Outputs 1 if value in input is above threshold, 0 otherwise.
  The defult threshold = 0, which means positive values would become 1 and 
  negative or 0, would become 0

  y = 1 if x > threshold
  y = 0 if x <= threshold
  
  y' = don't differenciable
*/
template <typename Dtype>
class ThresholdLayer : public NeuronLayer<Dtype> {
 public:
  explicit ThresholdLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_THRESHOLD;
  }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
    NOT_IMPLEMENTED;
  }

  Dtype threshold_;
};

}  // namespace caffe

#endif  // CAFFE_NEURON_LAYERS_HPP_
