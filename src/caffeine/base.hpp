#ifndef CAFFEINE_BASE_H_
#define CAFFEINE_BASE_H_

#include <vector>
#include "caffeine/blob.hpp"
#include "caffeine/common.hpp"
#include "caffeine/proto/layer_param.pb.h"

using std::vector;

namespace caffeine {

template <typename Dtype>
class Layer {
 public:
  explicit Layer(const LayerParameter& param)
    : layer_param_(param) {};
  virtual ~Layer();
  // SetUp: your function should implement this.
  virtual void SetUp(vector<const Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) = 0;

  // Forward, backward and predict wrappers. You should implement the cpu and
  // gpu specific implementations instead, and should not change these
  // functions.
  inline void Forward(vector<const Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  inline void Backward(vector<Blob<Dtype>*>& bottom,
      vector<const Blob<Dtype>*>* top, bool propagate_down);
  inline void Predict(vector<const Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  // The protobuf that stores the layer parameters
  LayerParameter layer_param_;
  // The vector that stores the parameters as a set of blobs.
  vector<Blob<Dtype> > blobs;

  // Forward functions
  virtual void Forward_cpu(vector<const Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) = 0;
  // If no gpu code is provided, we will simply use cpu code.
  virtual void Forward_gpu(vector<const Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
    LOG(WARNING) << "Using CPU code as backup.";
    Forward_cpu(bottom, top);
  };

  // Backward functions
  virtual void Backward_cpu(vector<Blob<Dtype>*>& bottom,
      vector<const Blob<Dtype>*>* top, bool propagate_down) = 0;
  virtual void Backward_gpu(vector<Blob<Dtype>*>& bottom,
      vector<const Blob<Dtype>*>* top, bool propagate_down) {
    LOG(WARNING) << "Using CPU code as backup.";
    Backward_cpu(bottom, top, propagate_down);
  };

  // Prediction functions: could be overridden, but the default behavior is to
  // simply call the forward functions.
  virtual void Predict_cpu(vector<const Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { Forward_cpu(bottom, top); };
  // For prediction, if there is no Predict_gpu, then there are two options:
  // to use predict_cpu as a backup, or to use forward_gpu (e.g. maybe the
  // author forgot to write what backup s/he wants?). Thus, we will require
  // the author to explicitly specify which fallback s/he wants.
  virtual void Predict_gpu(vector<const Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) = 0;
};  // class Layer

}  // namespace caffeine

#endif  // CAFFEINE_BASE_H_
