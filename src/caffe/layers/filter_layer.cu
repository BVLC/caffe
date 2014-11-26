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
  for (size_t b = 1; b < bottom.size(); b++) {
    const Dtype* bottom_data = bottom[b]->gpu_data();
    Dtype* top_data = top[b-1]->mutable_gpu_data();
    size_t dim = bottom[b]->count()/bottom[b]->num();

    for (size_t n = 0; n < new_tops_num; n++) {
      int offset = indices_to_forward_[n];
      int data_offset_top = dim*n;
      int data_offset_bottom = dim*offset;

      caffe_copy(dim, bottom_data+data_offset_bottom,
          top_data+data_offset_top);
    }
  }
}

template <typename Dtype>
void FilterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (size_t i = 1; i < propagate_down.size(); i++) {
    // bottom[0] is the selector and never needs backpropagation
    // so we can start from i = 1 and index each top with i-1
    if (propagate_down[i] && need_back_prop_[i-1]) {
      const int dim = top[i-1]->count()/top[i-1]->num();
      int index_top = 0;
      std::vector<double> zeros(dim, 0.0);
      for (size_t n = 0; n < bottom[i]->num(); n++) {
        int offset = indices_to_forward_[n];
        int data_offset_bottom = dim*n;
        if (n != offset) {  // this data was not been forwarded
          caffe_copy(dim, reinterpret_cast<Dtype*>(&zeros[0]),
              bottom[i]->mutable_gpu_diff() + data_offset_bottom);
        } else {  // this data was been forwarded
          int data_offset_top = dim*index_top;
          index_top++;
          caffe_copy(dim, top[i-1]->mutable_gpu_diff() + data_offset_top,
              bottom[i]->mutable_gpu_diff() + data_offset_bottom);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FilterLayer);

}  // namespace caffe
