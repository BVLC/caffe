#include <cmath>
#include <vector>

#include "caffe/layers/swish_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

template<typename Dtype, typename MItype, typename MOtype>
void SwishLayer<Dtype, MItype, MOtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype, MItype, MOtype>::LayerSetUp(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void SwishLayer<Dtype, MItype, MOtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype, MItype, MOtype>::Reshape(bottom, top);

  if (Caffe::mode() == Caffe::GPU && this->device_program_.get() == nullptr) {
    this->GenerateProgram();
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void SwishLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const MItype* bottom_data = bottom[0]->cpu_data();
  MOtype* top_data = top[0]->mutable_cpu_data();
  const int_tp count = bottom[0]->count();
  Dtype beta = this->layer_param_.swish_param().beta();
  for (int_tp i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i] *
        sigmoid<Dtype, MItype, MOtype>(beta * bottom_data[i]);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void SwishLayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    const MItype* bottom_data = bottom[0]->cpu_data();
    const MOtype* top_data = top[0]->cpu_data();
    const MOtype* top_diff = top[0]->cpu_diff();
    MItype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int_tp count = bottom[0]->count();
    Dtype beta = this->layer_param_.swish_param().beta();
    for (int_tp i = 0; i < count; ++i) {
      const Dtype swish_x = top_data[i];
      bottom_diff[i] = top_diff[i] * (beta * swish_x +
          sigmoid<Dtype, MItype, MOtype>(beta * bottom_data[i]) *
          (1. - beta * swish_x));
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SwishLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(SwishLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(SwishLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(SwishLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(Swish);
REGISTER_LAYER_CLASS_INST(Swish, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(Swish, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(Swish, (double), (double), (double));

}  // namespace caffe
