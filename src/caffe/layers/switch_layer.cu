#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void SwitchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int selector_ind = bottom.size() - 1;
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int num_elem = top[0]->channels() * top[0]->height() * top[0]->width();

  for (int n = 0; n < bottom[selector_ind]->num(); n++) {
    int index = static_cast<int>(bottom[selector_ind]->data_at(n, 0 , 0, 0));
    if (index >= 0 && index < selector_ind) {
      const Dtype* bottom_data = bottom[index]->gpu_data();
      caffe_copy(num_elem, bottom_data+bottom[index]->offset(n),
            top_data+top[0]->offset(n));
    } else {
      caffe_set(num_elem, Dtype(0), top_data+top[0]->offset(n));
    }
  }
}

template <typename Dtype>
void SwitchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int selector_ind = bottom.size() - 1;
  const int num_elem = top[0]->channels() * top[0]->height() * top[0]->width();
  const Dtype* top_diff = top[0]->gpu_diff();

  for (int n = 0; n < bottom[selector_ind]->num(); n++) {
    int index = static_cast<int>(bottom[selector_ind]->data_at(n, 0 , 0, 0));
    if (index >= 0 && index < selector_ind) {
      Dtype* bottom_diff = bottom[index]->mutable_gpu_diff();
      caffe_copy(num_elem, top_diff+top[0]->offset(n),
          bottom_diff + bottom[index]->offset(n));
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SwitchLayer);

}  // namespace caffe
