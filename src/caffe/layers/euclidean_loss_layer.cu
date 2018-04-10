#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void EuclideanLossLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  int_tp count = bottom[0]->count();
  Dtype dot;

  this->device_->template sub<Dtype>(count, bottom[0]->gpu_data(),
                               bottom[1]->gpu_data(), diff_.mutable_gpu_data());
  // Scale the error element-wise
  if (bottom.size() == 3) {
    this->device_->template mul<Dtype>(count, diff_.mutable_gpu_data(),
                               bottom[2]->gpu_data(), diff_.mutable_gpu_data());
  }
  this->device_->template dot<Dtype>(count, diff_.gpu_data(), diff_.gpu_data(),
                                     &dot);

  Dtype loss = dot / static_cast<Dtype>(bottom[0]->count(0)) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template<typename Dtype, typename MItype, typename MOtype>
void EuclideanLossLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  for (int_tp i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0]
          / static_cast<Dtype>(bottom[0]->count(0));
        this->device_->template axpby<Dtype>(bottom[i]->count(), // count
                             alpha,                              // alpha
                             diff_.gpu_data(),                   // a
                             Dtype(0),                           // beta
                             bottom[i]->mutable_gpu_diff());     // b
    }
  }
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(EuclideanLossLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(EuclideanLossLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(EuclideanLossLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(EuclideanLossLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(EuclideanLossLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(EuclideanLossLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
