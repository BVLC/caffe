#include <vector>

#include "caffe/layers/tanh_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void TanHLayer<Dtype, MItype, MOtype>::Reshape(
      const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
  NeuronLayer<Dtype, MItype, MOtype>::Reshape(bottom, top);

  if (Caffe::mode() == Caffe::GPU && this->device_program_.get() == nullptr) {
    this->GenerateProgram();
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void TanHLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int_tp count = bottom[0]->count();
  for (int_tp i = 0; i < count; ++i) {
    top_data[i] = tanh(bottom_data[i]);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void TanHLayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int_tp count = bottom[0]->count();
    Dtype tanhx;
    for (int_tp i = 0; i < count; ++i) {
      tanhx = top_data[i];
      bottom_diff[i] = top_diff[i] * (1 - tanhx * tanhx);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TanHLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(TanHLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(TanHLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(TanHLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(TanH);
REGISTER_LAYER_CLASS_INST(TanH, (half_fp), (half_fp), (half_fp));

}  // namespace caffe
