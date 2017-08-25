#include <algorithm>
#include <vector>

#include "caffe/layers/bnll_layer.hpp"

namespace caffe {

const float kBNLL_THRESHOLD = 50.;

template <typename Dtype>
void BNLLLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i] > 0 ?
        bottom_data[i] + log(1. + exp(-bottom_data[i])) :
        log(1. + exp(bottom_data[i]));
  }
}

#ifdef CPU_ONLY
STUB_GPU(BNLLLayer);
#endif

INSTANTIATE_CLASS(BNLLLayer);
REGISTER_LAYER_CLASS(BNLL);

}  // namespace caffe
