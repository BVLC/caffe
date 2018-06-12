#include <algorithm>
#include <vector>

#include "caffe/layers/bnll_layer.hpp"

namespace caffe {

const float kBNLL_THRESHOLD = 50.;


template<typename Dtype, typename MItype, typename MOtype>
void BNLLLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int_tp count = bottom[0]->count();
  for (int_tp i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i] > 0 ?
        bottom_data[i] + std::log(1. + std::exp(-bottom_data[i])) :
        std::log(1. + std::exp(bottom_data[i]));
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void BNLLLayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int_tp count = bottom[0]->count();
    Dtype expval;
    for (int_tp i = 0; i < count; ++i) {
      expval = std::exp(std::min(bottom_data[i], Dtype(kBNLL_THRESHOLD)));
      bottom_diff[i] = top_diff[i] * expval / (expval + 1.);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BNLLLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(BNLLLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(BNLLLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(BNLLLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(BNLL);
REGISTER_LAYER_CLASS_INST(BNLL, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(BNLL, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(BNLL, (double), (double), (double));

}  // namespace caffe
