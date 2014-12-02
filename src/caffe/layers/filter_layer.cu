#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void FilterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int new_tops_num = indices_to_forward_.size();
  // forward all filtered items for all bottoms but the Selector (bottom[0])
  for (int b = 1; b < bottom.size(); b++) {
    const Dtype* bottom_data = bottom[b]->gpu_data();
    Dtype* top_data = top[b-1]->mutable_gpu_data();
    int dim = bottom[b]->count() / bottom[b]->num();

    for (int n = 0; n < new_tops_num; n++) {
      int data_offset_top = top[b-1]->offset(n);
      int data_offset_bottom =  bottom[b]->offset(indices_to_forward_[n]);

      caffe_copy(dim, bottom_data+data_offset_bottom,
          top_data+data_offset_top);
    }
  }
}

template <typename Dtype>
void FilterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 1; i < propagate_down.size(); i++) {
    // bottom[0] is the selector and never needs backpropagation
    // so we can start from i = 1 and index each top with i-1
    if (propagate_down[i] && need_back_prop_[i-1]) {
      const int dim = top[i-1]->count() / top[i-1]->num();
      int next_to_backward_offset = 0;
      int batch_offset = 0;
      int data_offset_bottom = 0;
      int data_offset_top = 0;
      for (int n = 0; n < bottom[i]->num(); n++) {
        if (next_to_backward_offset >= indices_to_forward_.size()) {
          // we already visited all items that were been forwarded, so
          // just set to zero remaining ones
          data_offset_bottom = top[i-1]->offset(n);
          caffe_gpu_set(dim, Dtype(0),
              bottom[i]->mutable_gpu_diff() + data_offset_bottom);
        } else {
          batch_offset = indices_to_forward_[next_to_backward_offset];
          data_offset_bottom = top[i-1]->offset(n);
          if (n != batch_offset) {  // this data was not been forwarded
            caffe_gpu_set(dim, Dtype(0),
                bottom[i]->mutable_gpu_diff() + data_offset_bottom);
          } else {  // this data was been forwarded
            data_offset_top = top[i-1]->offset(next_to_backward_offset);
            next_to_backward_offset++;  // point to next forwarded item index
            caffe_copy(dim, top[i-1]->mutable_gpu_diff() + data_offset_top,
                bottom[i]->mutable_gpu_diff() + data_offset_bottom);
          }
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FilterLayer);

}  // namespace caffe
