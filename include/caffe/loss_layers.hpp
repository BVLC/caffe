// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_LOSS_LAYERS_HPP_
#define CAFFE_LOSS_LAYERS_HPP_

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
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// LossLayer takes two inputs of same num, and has no output.
template <typename Dtype>
class LossLayer : public Layer<Dtype> {
 public:
  explicit LossLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void SetUp(
      const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top);
  virtual void FurtherSetUp(
      const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top);
};

// SigmoidCrossEntropyLossLayer
template <typename Dtype>
class SigmoidCrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
  explicit SigmoidCrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
          sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
          sigmoid_output_(new Blob<Dtype>()) {}
  virtual void FurtherSetUp(const vector<Blob<Dtype>*>& bottom,
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

  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  // sigmoid_output stores the output of the sigmoid layer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  // Vector holders to call the underlying sigmoid layer forward and backward.
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  vector<Blob<Dtype>*> sigmoid_top_vec_;
};

// EuclideanLossLayer: compute y = 1/2 \sum_i (x_i - x'_i)^2
template <typename Dtype>
class EuclideanLossLayer : public LossLayer<Dtype> {
 public:
  explicit EuclideanLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), difference_() {}
  virtual void FurtherSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> difference_;
};


template <typename Dtype>
class InfogainLossLayer : public LossLayer<Dtype> {
 public:
  explicit InfogainLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), infogain_() {}
  virtual void FurtherSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> infogain_;
};

template <typename Dtype>
class HingeLossLayer : public LossLayer<Dtype> {
 public:
  explicit HingeLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

template <typename Dtype>
class MultinomialLogisticLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultinomialLogisticLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void FurtherSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

// AccuracyLayer: not an actual loss layer;
// computes the accuracy and logprob of x with respect to y'.
template <typename Dtype>
class AccuracyLayer : public Layer<Dtype> {
 public:
  explicit AccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  // The accuracy layer should not be used to compute backward operations.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
    NOT_IMPLEMENTED;
  }
};

}  // namespace caffe

#endif  // CAFFE_LOSS_LAYERS_HPP_
