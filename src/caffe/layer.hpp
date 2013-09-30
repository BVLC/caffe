// Copyright 2013 Yangqing Jia

#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

using std::vector;

namespace caffe {

template <typename Dtype>
class Layer {
 public:
  // You should not implement your own constructor. Any set up code should go
  // to SetUp(), where the dimensions of the bottom blobs are provided to the
  // layer.
  explicit Layer(const LayerParameter& param)
    : layer_param_(param) {}
  virtual ~Layer() {}
  // SetUp: your function should implement this.
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) = 0;

  // Forward and backward wrappers. You should implement the cpu and
  // gpu specific implementations instead, and should not change these
  // functions.
  inline void Forward(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  inline Dtype Backward(const vector<Blob<Dtype>*>& top,
      const bool propagate_down,
      vector<Blob<Dtype>*>* bottom);

  // Returns the vector of parameters.
  vector<shared_ptr<Blob<Dtype> > >& params() {
    return blobs_;
  }

  // Writes the layer parameter to a protocol buffer
  virtual void ToProto(LayerParameter* param, bool write_diff = false);

 protected:
  // The protobuf that stores the layer parameters
  LayerParameter layer_param_;
  // The vector that stores the parameters as a set of blobs.
  vector<shared_ptr<Blob<Dtype> > > blobs_;

  // Forward functions
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) = 0;
  // If no gpu code is provided, we will simply use cpu code.
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
    LOG(WARNING) << "Using CPU code as backup.";
    Forward_cpu(bottom, top);
  };

  // Backward functions: the backward function will compute the gradients for
  // any parameters and also for the bottom blobs if propagate_down is true.
  // It will return the loss produced from this layer.
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down,
      vector<Blob<Dtype>*>* bottom) = 0;
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down,
      vector<Blob<Dtype>*>* bottom) {
    LOG(WARNING) << "Using CPU code as backup.";
    return Backward_cpu(top, propagate_down, bottom);
  };

  DISABLE_COPY_AND_ASSIGN(Layer);
};  // class Layer

// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
inline void Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Forward_cpu(bottom, top);
    break;
  case Caffe::GPU:
    Forward_gpu(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
};

template <typename Dtype>
inline Dtype Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return Backward_cpu(top, propagate_down, bottom);
  case Caffe::GPU:
    return Backward_gpu(top, propagate_down, bottom);
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
};

template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (int i = 0; i < blobs_.size(); ++i) {
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  }
}

}  // namespace caffe

#endif  // CAFFE_LAYER_H_
