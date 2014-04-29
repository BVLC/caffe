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

// NeuronLayer is an interface for layers that work on single elements.
template <typename Dtype>
class NeuronLayer : public Layer<Dtype> {
 public:
  explicit NeuronLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
};

// BNLLLayer
template <typename Dtype>
class BNLLLayer : public NeuronLayer<Dtype> {
 public:
  explicit BNLLLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

// DropoutLayer sets random portion of its input to 0.
template <typename Dtype>
class DropoutLayer : public NeuronLayer<Dtype> {
 public:
  explicit DropoutLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  shared_ptr<SyncedMemory> rand_vec_;
  Dtype threshold_;
  Dtype scale_;
  unsigned int uint_thres_;
};

// PowerLayer computes y = (shift + scale * x)^power
template <typename Dtype>
class PowerLayer : public NeuronLayer<Dtype> {
 public:
  explicit PowerLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  Dtype power_;
  Dtype scale_;
  Dtype shift_;
  Dtype diff_scale_;
};

// ReLULayer computes y = max(0, x).
template <typename Dtype>
class ReLULayer : public NeuronLayer<Dtype> {
 public:
  explicit ReLULayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

// SigmoidLayer computes y = 1. / (1 + exp(-x))
template <typename Dtype>
class SigmoidLayer : public NeuronLayer<Dtype> {
 public:
  explicit SigmoidLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

// TanHLayer: computes y = 1. * (exp(2 * x) - 1) / (exp(2 * x) + 1)
template <typename Dtype>
class TanHLayer : public NeuronLayer<Dtype> {
 public:
  explicit TanHLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

}  // namespace caffe

#endif  // CAFFE_NEURON_LAYERS_HPP_
