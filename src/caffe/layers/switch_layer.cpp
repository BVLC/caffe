#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// Let's move the selector to the last bottom
template <typename Dtype>
void SwitchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Check that the dimensions of bottoms are all the same
  for (int i = 0; i < bottom.size() - 1; ++i) {
    CHECK_EQ(bottom[i]->num(),  bottom[0]->num());
    CHECK_EQ(bottom[i]->channels(), bottom[0]->channels());
    CHECK_EQ(bottom[i]->height(), bottom[0]->height());
    CHECK_EQ(bottom[i]->width(), bottom[0]->width());
  }
  // Check the selector dimensions
  // It could be generalized to have more channels, one per top
  const int selector_ind = bottom.size() - 1;
  CHECK_EQ(bottom[selector_ind]->num(), bottom[0]->num());
  CHECK_EQ(bottom[selector_ind]->channels(), 1);
  CHECK_EQ(bottom[selector_ind]->height(), 1);
  CHECK_EQ(bottom[selector_ind]->width(), 1);
}

template <typename Dtype>
void SwitchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialize with the first blob.
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
    bottom[0]->height(), bottom[0]->width());
  CHECK_EQ(bottom[0]->count(), top[0]->count());
}

template <typename Dtype>
void SwitchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int selector_ind = bottom.size() - 1;
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num_elem = top[0]->channels() * top[0]->height() * top[0]->width();

  for (int n = 0; n < bottom[selector_ind]->num(); n++) {
    int index = static_cast<int>(bottom[selector_ind]->data_at(n, 0 , 0, 0));
    if (index >= 0 && index < selector_ind) {
      const Dtype* bottom_data = bottom[index]->cpu_data();
      caffe_copy(num_elem, bottom_data+bottom[index]->offset(n),
            top_data+top[0]->offset(n));
    } else {
      caffe_set(num_elem, Dtype(0), top_data+top[0]->offset(n));
    }
  }
}

template <typename Dtype>
void SwitchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int selector_ind = bottom.size() - 1;
  const int num_elem = top[0]->channels() * top[0]->height() * top[0]->width();
  const Dtype* top_diff = top[0]->cpu_diff();

  for (int n = 0; n < bottom[selector_ind]->num(); n++) {
    int index = static_cast<int>(bottom[selector_ind]->data_at(n, 0 , 0, 0));
    if (index >= 0 && index < selector_ind) {
      Dtype* bottom_diff = bottom[index]->mutable_cpu_diff();
      caffe_copy(num_elem, top_diff+top[0]->offset(n),
          bottom_diff + bottom[index]->offset(n));
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SwitchLayer);
#endif

INSTANTIATE_CLASS(SwitchLayer);
REGISTER_LAYER_CLASS(SWITCH, SwitchLayer);
}  // namespace caffe
