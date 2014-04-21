// Copyright 2014 BVLC and contributors.

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
    : layer_param_(param) {
      // The only thing we do is to copy blobs if there are any.
      if (layer_param_.blobs_size() > 0) {
        blobs_.resize(layer_param_.blobs_size());
        for (int i = 0; i < layer_param_.blobs_size(); ++i) {
          blobs_[i].reset(new Blob<Dtype>());
          blobs_[i]->FromProto(layer_param_.blobs(i));
        }
      }
    }
  virtual ~Layer() {}
  // SetUp: your function should implement this.
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) = 0;

  // Forward and backward wrappers. You should implement the cpu and
  // gpu specific implementations instead, and should not change these
  // functions.
  inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  inline void Backward(const vector<Blob<Dtype>*>& top,
      const bool propagate_down,
      vector<Blob<Dtype>*>* bottom);

  // Returns the vector of blobs.
  vector<shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }

  // Returns the layer parameter
  const LayerParameter& layer_param() { return layer_param_; }
  // Writes the layer parameter to a protocol buffer
  virtual void ToProto(LayerParameter* param, bool write_diff = false);

 protected:
  // The protobuf that stores the layer parameters
  LayerParameter layer_param_;
  // The vector that stores the parameters as a set of blobs.
  vector<shared_ptr<Blob<Dtype> > > blobs_;

  // Forward functions: compute the layer output
  // (and loss layers return the loss; other layers return the dummy value 0.)
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) = 0;
  // If no gpu code is provided, we will simply use cpu code.
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    return Forward_cpu(bottom, top);
  }

  // Backward functions: compute the gradients for any parameters and
  // for the bottom blobs if propagate_down is true.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down,
      vector<Blob<Dtype>*>* bottom) = 0;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down,
      vector<Blob<Dtype>*>* bottom) {
    // LOG(WARNING) << "Using CPU code as backup.";
    Backward_cpu(top, propagate_down, bottom);
  }

  DISABLE_COPY_AND_ASSIGN(Layer);
};  // class Layer

// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return Forward_cpu(bottom, top);
  case Caffe::GPU:
    return Forward_gpu(bottom, top);
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return Dtype(0);
  }
}

template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Backward_cpu(top, propagate_down, bottom);
    break;
  case Caffe::GPU:
    Backward_gpu(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

// Serialize LayerParameter to protocol buffer
template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (int i = 0; i < blobs_.size(); ++i) {
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  }
}

// The layer factory function
template <typename Dtype>
Layer<Dtype>* GetLayer(const LayerParameter& param);

}  // namespace caffe

#endif  // CAFFE_LAYER_H_
