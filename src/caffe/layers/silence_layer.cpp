#include <vector>

#include "caffe/layers/silence_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void SilenceLayer<Dtype, MItype, MOtype>::Backward_cpu(
                                        const vector<Blob<MOtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<MItype>*>& bottom) {
  for (int_tp i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_set(bottom[i]->count(), Dtype(0),
                bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SilenceLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(SilenceLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(SilenceLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(SilenceLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(Silence);
REGISTER_LAYER_CLASS_INST(Silence, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(Silence, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(Silence, (double), (double), (double));

}  // namespace caffe
