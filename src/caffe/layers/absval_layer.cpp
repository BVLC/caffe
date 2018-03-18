#include <vector>

#include "caffe/layers/absval_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void AbsValLayer<Dtype, MItype, MOtype>::LayerSetUp(
      const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
  NeuronLayer<Dtype, MItype, MOtype>::LayerSetUp(bottom, top);
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";

  this->InitializeQuantizers(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void AbsValLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const int_tp count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_abs(count, bottom[0]->cpu_data(), top_data);
}

template<typename Dtype, typename MItype, typename MOtype>
void AbsValLayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  const int_tp count = top[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sign(count, bottom_data, bottom_diff);
    caffe_mul(count, bottom_diff, top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(AbsValLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(AbsValLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(AbsValLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(AbsValLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(AbsVal);
REGISTER_LAYER_CLASS_INST(AbsVal, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(AbsVal, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(AbsVal, (double), (double), (double));

}  // namespace caffe
