#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void ReLULayer<Dtype, MItype, MOtype>::Reshape(
      const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
  NeuronLayer<Dtype, MItype, MOtype>::Reshape(bottom, top);

  if (Caffe::mode() == Caffe::GPU && this->device_program_.get() == nullptr) {
    this->GenerateProgram();
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void ReLULayer<Dtype, MItype, MOtype>::Forward_cpu(
                                     const vector<Blob<MItype>*>& bottom,
                                     const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int_tp count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int_tp i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void ReLULayer<Dtype, MItype, MOtype>::Backward_cpu(
                                     const vector<Blob<MOtype>*>& top,
                                     const vector<bool>& propagate_down,
                                     const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int_tp count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int_tp i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS_3T(ReLULayer);

}  // namespace caffe
