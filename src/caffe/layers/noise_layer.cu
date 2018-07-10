#include <vector>

#include "caffe/layers/noise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void NoiseLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  if (this->phase_ == caffe::TEST) {
    this->device_->template set<Dtype>(bottom[0]->count(),
                                        Dtype(0), top_data);
  } else {
    Dtype* noise_data = noise_.mutable_cpu_data();
    caffe_rng_gaussian<Dtype>(bottom[0]->shape()[0],
                              this->layer_param().noise_param().mu(),
                              this->layer_param().noise_param().sigma(),
                              noise_data);
    for (size_t i = 0; i < bottom[0]->shape()[0]; ++i) {
      this->device_->template scale<Dtype>(
          bottom[0]->count(1), noise_data[i],
          bottom_data + i * bottom[0]->count(1),
          top_data + i * bottom[0]->count(1));
    }
  }
}


template<typename Dtype, typename MItype, typename MOtype>
void NoiseLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                      const vector<Blob<MOtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    vptr<const Dtype> top_diff = top[0]->gpu_diff();
    vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == caffe::TEST) {
      this->device_->template set<Dtype>(bottom[0]->count(),
                                          Dtype(0), bottom_diff);
    } else {
      const Dtype* noise_data = noise_.cpu_data();
      for (size_t i = 0; i < bottom[0]->shape()[0]; ++i) {
        this->device_->template scale<Dtype>(
            bottom[0]->count(1), noise_data[i],
            top_diff + i * bottom[0]->count(1),
            bottom_diff + i * bottom[0]->count(1));
      }
    }
  }
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(NoiseLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(NoiseLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(NoiseLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(NoiseLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(NoiseLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(NoiseLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
