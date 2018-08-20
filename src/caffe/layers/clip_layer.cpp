#include <algorithm>
#include <vector>

#include "caffe/layers/clip_layer.hpp"

namespace caffe {

template <typename Dtype>
void ClipLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();

  Dtype min = this->layer_param_.clip_param().min();
  Dtype max = this->layer_param_.clip_param().max();

  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(min, std::min(bottom_data[i], max));
  }
}

template <typename Dtype>
void ClipLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();

    Dtype min = this->layer_param_.clip_param().min();
    Dtype max = this->layer_param_.clip_param().max();

    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * (
              bottom_data[i] >= min && bottom_data[i] <= max);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ClipLayer);
#endif

INSTANTIATE_CLASS(ClipLayer);
REGISTER_LAYER_CLASS(Clip);

}  // namespace caffe
