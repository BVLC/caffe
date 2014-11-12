#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SwitchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SwitchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* select = bottom[0]->cpu_data();
  int num_elem = bottom[0]->count();
  for (int i = 0; i < num_elem; i++) {
    top_data[i] = bottom[select[i]+1]->cpu_data()[i];
  }
}

template <typename Dtype>
void SwitchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // don't need to implement yet
}

#ifdef CPU_ONLY
STUB_GPU(SwitchLayer);
#endif

INSTANTIATE_CLASS(SwitchLayer);
//REGISTER_LAYER_CLASS(SWITCH, SwitchLayer);
}  // namespace caffe
