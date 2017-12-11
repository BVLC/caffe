#include <vector>

#include "caffe/layers/absval_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AbsValLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_const_gpu(bottom,top);
}

template <typename Dtype>
void AbsValLayer<Dtype>::Forward_const_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) const {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_abs(count, bottom[0]->gpu_data(), top_data);
}

INSTANTIATE_LAYER_GPU_FUNCS(AbsValLayer);


}  // namespace caffe
