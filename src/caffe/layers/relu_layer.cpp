#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    const int count = bottom[i]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int j = 0; j < count; ++j) {
      top_data[j] = std::max(bottom_data[j], Dtype(0))
          + negative_slope * std::min(bottom_data[j], Dtype(0));
    }
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      const Dtype* top_diff = top[i]->cpu_diff();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      const int count = bottom[i]->count();
      Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
      for (int j = 0; j < count; ++j) {
        bottom_diff[j] = top_diff[j] * ((bottom_data[j] > 0)
            + negative_slope * (bottom_data[j] <= 0));
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
