#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void DotProductLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data_a = bottom[0]->gpu_data();
  const Dtype* bottom_data_b = bottom[1]->gpu_data();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  Dtype* top_data = top[0]->mutable_gpu_data();

  for (int i = 0; i < num; ++i) {
    caffe_gpu_dot(dim, bottom_data_a + i*dim, bottom_data_b + i*dim, top_data + i);
  }
}

template <typename Dtype>
void DotProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();

  const Dtype* bottom_data_a = bottom[0]->gpu_data();
  const Dtype* bottom_data_b = bottom[1]->gpu_data();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;

  if (propagate_down[0]){
    Dtype* bottom_diff_a = bottom[0]->mutable_gpu_diff();
    for (int i = 0; i < num; ++i) {
      caffe_gpu_scale(dim, top_diff[i], bottom_data_b + i*dim, bottom_diff_a + i*dim);
    }
  }

  if (propagate_down[1]){
    Dtype* bottom_diff_b = bottom[1]->mutable_gpu_diff();
    for (int i = 0; i < num; ++i) {
      caffe_gpu_scale(dim, top_diff[i], bottom_data_a + i*dim, bottom_diff_b + i*dim);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DotProductLayer);


}  // namespace caffe
