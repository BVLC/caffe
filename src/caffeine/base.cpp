#include "caffeine/base.hpp"

namespace caffeine {

// Forward, backward and predict wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
inline void Layer<Dtype>::Forward(vector<const Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  switch(Caffeine::mode()) {
  case Caffeine::CPU:
    Forward_cpu(bottom, top);
    break;
  case Caffeine::GPU:
    Forward_gpu(bottom, top);
    break;
  default:
    CHECK(false);
  }
};

template <typename Dtype>
inline void Layer<Dtype>::Backward(vector<Blob<Dtype>*>& bottom,
    vector<const Blob<Dtype>*>* top, bool propagate_down) {
  switch(Caffeine::mode()) {
  case Caffeine::CPU:
    Backward_cpu(bottom, top, propagate_down);
    break;
  case Caffeine::GPU:
    Backward_gpu(bottom, top, propagate_down);
    break;
  default:
    CHECK(false);
  }
};

template <typename Dtype>
inline void Layer<Dtype>::Predict(vector<const Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  switch(Caffeine::mode()) {
  case Caffeine::CPU:
    Predict_cpu(bottom, top);
    break;
  case Caffeine::GPU:
    Predict_gpu(bottom, top);
    break;
  default:
    CHECK(false);
  }
};

}  // namespace caffeine
