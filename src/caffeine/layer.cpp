#include "caffeine/layer.hpp"

namespace caffeine {

// Forward, backward and predict wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
inline void Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  switch(Caffeine::mode()) {
  case Caffeine::CPU:
    Forward_cpu(bottom, top);
    break;
  case Caffeine::GPU:
    Forward_gpu(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown caffeine mode.";
  }
};

template <typename Dtype>
inline Dtype Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  switch(Caffeine::mode()) {
  case Caffeine::CPU:
    return Backward_cpu(top, propagate_down, bottom);
    break;
  case Caffeine::GPU:
    return Backward_gpu(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown caffeine mode.";
  }
};

template class Layer<float>;
template class Layer<double>;

}  // namespace caffeine
