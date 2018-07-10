#include <vector>

#include "caffe/layers/absval_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void AbsValLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const int_tp count = top[0]->count();
  this->device_->template abs<Dtype>(count, bottom[0]->gpu_data(),
                                     top[0]->mutable_gpu_data());
}

template<typename Dtype, typename MItype, typename MOtype>
void AbsValLayer<Dtype, MItype, MOtype>::Backward_gpu(
                            const vector<Blob<MOtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<MItype>*>& bottom) {
  const int_tp count = top[0]->count();
  vptr<const MOtype> top_diff = top[0]->gpu_diff();
  if (propagate_down[0]) {
    vptr<const MItype> bottom_data = bottom[0]->gpu_data();
    vptr<MItype> bottom_diff = bottom[0]->mutable_gpu_diff();

    this->device_->template sign<Dtype>(count, bottom_data, bottom_diff);
    this->device_->template mul<Dtype>(count, bottom_diff, top_diff,
                                       bottom_diff);
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(AbsValLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(AbsValLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(AbsValLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(AbsValLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(AbsValLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(AbsValLayer, Backward_gpu,
                                  (double), (double), (double));


}  // namespace caffe
