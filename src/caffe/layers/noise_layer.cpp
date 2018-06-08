#include <vector>

#include "caffe/layers/noise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void NoiseLayer<Dtype, MItype, MOtype>::LayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {

  this->InitializeQuantizers(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void NoiseLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  std::vector<int_tp> noise_shape(1, bottom[0]->shape()[0]);
  this->noise_.Reshape(noise_shape);
  top[0]->ReshapeLike(*bottom[0]);
}

template<typename Dtype, typename MItype, typename MOtype>
void NoiseLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  if (this->phase_ == caffe::TEST) {
    caffe_set(bottom[0]->count(), Dtype(0), top_data);
  } else {
    Dtype* noise_data = noise_.mutable_cpu_data();
    caffe_rng_gaussian<Dtype>(bottom[0]->shape()[0],
                              this->layer_param().noise_param().mu(),
                              this->layer_param().noise_param().sigma(),
                              noise_data);
    for (size_t i = 0; i < bottom[0]->shape()[0]; ++i) {
      caffe_scale(bottom[0]->count(1), noise_data[i],
                  bottom_data + i * bottom[0]->count(1),
                  top_data + i * bottom[0]->count(1));
    }
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void NoiseLayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == caffe::TEST) {
      caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    } else {
      const Dtype* noise_data = noise_.cpu_data();
      for (size_t i = 0; i < bottom[0]->shape()[0]; ++i) {
        caffe_scale(bottom[0]->count(1), noise_data[i],
                    top_diff + i * bottom[0]->count(1),
                    bottom_diff + i * bottom[0]->count(1));
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(NoiseLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(NoiseLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(NoiseLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(NoiseLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(Noise);
REGISTER_LAYER_CLASS_INST(Noise, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(Noise, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(Noise, (double), (double), (double));


}  // namespace caffe
