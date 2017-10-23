#include <vector>

#include "caffe/layers/silence_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void SilenceLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                        const vector<Blob<MItype>*>& bottom,
                                        const vector<Blob<MOtype>*>& top) {
  // Do nothing.
}

template<typename Dtype, typename MItype, typename MOtype>
void SilenceLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                        const vector<Blob<MOtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<MItype>*>& bottom) {
  for (int_tp i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      this->device_->set(bottom[i]->count(), Dtype(0),
                         bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SilenceLayer);

}  // namespace caffe
