#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

template<typename Dtype, typename MItype, typename MOtype>
void SigmoidLayer<Dtype, MItype, MOtype>::Reshape(
                                        const vector<Blob<MItype>*>& bottom,
                                        const vector<Blob<MOtype>*>& top) {
  NeuronLayer<Dtype, MItype, MOtype>::Reshape(bottom, top);

  if (Caffe::mode() == Caffe::GPU && this->device_program_.get() == nullptr) {
    this->GenerateProgram();
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void SigmoidLayer<Dtype, MItype, MOtype>::Forward_cpu(
                                        const vector<Blob<MItype>*>& bottom,
                                        const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int_tp count = bottom[0]->count();
  for (int_tp i = 0; i < count; ++i) {
    top_data[i] = sigmoid<Dtype, MItype, MOtype>(bottom_data[i]);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void SigmoidLayer<Dtype, MItype, MOtype>::Backward_cpu(
                                        const vector<Blob<MOtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int_tp count = bottom[0]->count();
    for (int_tp i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i];
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(SigmoidLayer,
                     (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(SigmoidLayer,
                     (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(SigmoidLayer,
                     (double), (double), (double));


}  // namespace caffe
