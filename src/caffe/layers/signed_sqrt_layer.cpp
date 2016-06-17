#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/signed_sqrt_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SignedSqrtLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK(top[0] != bottom[0]) << "do not support in place operation.";
    top[0]->Reshape(bottom[0]->shape());
}

template <typename Dtype>
void SignedSqrtLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();

  for (int i = 0; i < count; i++) {
      if (bottom_data[i] >= 0)
        top_data[i] = sqrt(bottom_data[i]);
      else
        top_data[i] = -sqrt(-bottom_data[i]);
  }
}

template <typename Dtype>
void SignedSqrtLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int count = bottom[0]->count();

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* top_data = top[0]->cpu_data();

    caffe_abs(count, top_data, bottom_diff);
    caffe_add_scalar(count, epsilon, bottom_diff);
    caffe_div(count, top_diff, bottom_diff, bottom_diff);
    caffe_scal(count, Dtype(0.5), bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SignedSqrtLayer);
#endif

INSTANTIATE_CLASS(SignedSqrtLayer);
REGISTER_LAYER_CLASS(SignedSqrt);

}  // namespace caffe
