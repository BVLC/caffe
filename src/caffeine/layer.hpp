#ifndef CAFFEINE_LAYER_H_
#define CAFFEINE_LAYER_H_

#include <vector>
#include "caffeine/blob.hpp"
#include "caffeine/common.hpp"
#include "caffeine/proto/layer_param.pb.h"

using std::vector;

namespace caffeine {

template <typename Dtype>
class Layer {
 public:
   // You should not implement your own constructor. Any set up code should go
   // to SetUp(), where the dimensions of the bottom blobs are provided to the
   // layer.
  explicit Layer(const LayerParameter& param)
    : layer_param_(param) {};
  virtual ~Layer() {};
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

 protected:
  // The protobuf that stores the layer parameters
  LayerParameter layer_param_;
  // The vector that stores the parameters as a set of blobs.
  vector<Blob<Dtype> > blobs_;

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
};  // class Layer

}  // namespace caffeine

#endif  // CAFFEINE_LAYER_H_
