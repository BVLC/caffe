#include <algorithm>
#include <vector>

#include "caffe/layers/elu_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void ELULayer<Dtype, MItype, MOtype>::Reshape(
                            const vector<Blob<MItype>*>& bottom,
                            const vector<Blob<MOtype>*>& top) {
  NeuronLayer<Dtype, MItype, MOtype>::Reshape(bottom, top);

  if (Caffe::mode() == Caffe::GPU && this->device_program_.get() == nullptr) {
    this->GenerateProgram();
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void ELULayer<Dtype, MItype, MOtype>::Forward_cpu(
                                    const vector<Blob<MItype>*>& bottom,
                                    const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype alpha = this->layer_param_.elu_param().alpha();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + alpha * (std::exp(std::min(bottom_data[i], Dtype(0))) - Dtype(1));
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void ELULayer<Dtype, MItype, MOtype>::Backward_cpu(
                                    const vector<Blob<MOtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype alpha = this->layer_param_.elu_param().alpha();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + (alpha + top_data[i]) * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ELULayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(ELULayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(ELULayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(ELULayer, (double), (double), (double));

REGISTER_LAYER_CLASS(ELU);
REGISTER_LAYER_CLASS_INST(ELU, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(ELU, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(ELU, (double), (double), (double));

}  // namespace caffe
