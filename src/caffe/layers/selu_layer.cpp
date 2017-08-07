#include <algorithm>
#include <vector>

#include "caffe/layers/selu_layer.hpp"

namespace caffe {

template <typename Dtype>
void SeLuLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    alpha = this->layer_param_.selu_param().alpha();
    lambda = this->layer_param_.selu_param().lambda();
}

template <typename Dtype>
void SeLuLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i] > Dtype(0.) ? lambda*bottom_data[i] : 
                    lambda*alpha*(exp(bottom_data[i])-Dtype(1.));
  }
}

template <typename Dtype>
void SeLuLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = bottom_data[i] > 0 ? lambda*top_diff[i] : 
                        lambda*alpha*top_diff[i]*exp(bottom_data[i]);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SeLuLayer);
#endif

INSTANTIATE_CLASS(SeLuLayer);
REGISTER_LAYER_CLASS(SeLu);

}  // namespace caffe
