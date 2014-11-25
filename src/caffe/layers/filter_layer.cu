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
  for(size_t b = 1; b < bottom.size(); b++) {
    const Dtype* bottom_data = bottom[b]->gpu_data();
    Dtype* top_data = top[b-1]->mutable_gpu_data();
    size_t size_single_item = bottom[b]->count()/bottom[b]->num();

    for (size_t n = 0; n < new_tops_num; n++) {
      int offset = indices_to_forward_[n];
      int data_offset_top = size_single_item*n;
      int data_offset_bottom = size_single_item*offset;

      caffe_copy(size_single_batch, bottom_data+data_offset_bottom,
          top_data+data_offset_top);
    }
  }
}

template <typename Dtype>
void FilterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(ERROR) << "propagate_down.size(): " << propagate_down.size();
  LOG(ERROR) << "bottom.size(): " << bottom.size();
  LOG(ERROR) << "top.size(): " << top.size();
  for(size_t i = 0; i < propagate_down.size(); i++) {
    if (propagate_down[i]) {/*
      const int size_single_batch = top[1]->count()/top[1]->num();
      int index_top = 0;
      std::vector<double> zeros(size_single_batch, 0.0);
      for (size_t n = 0; n < bottom[1]->num(); n++) {
        int offset = indices_to_forward_[n];
        int data_offset_bottom = size_single_batch*n;
        if (n != offset) {  // this data was not been forwarded
          caffe_copy(size_single_batch,
              reinterpret_cast<Dtype*>(&zeros[0]),
              bottom[1]->mutable_gpu_diff() + data_offset_bottom);
        } else {  // this data was been forwarded
          int data_offset_top = size_single_batch*index_top;
          index_top++;
          caffe_copy(size_single_batch,
              top[1]->mutable_gpu_diff() + data_offset_top,
              bottom[1]->mutable_gpu_diff() + data_offset_bottom);
        }
      }
    */}
  }
    
}
INSTANTIATE_LAYER_GPU_FUNCS(FilterLayer);

}  // namespace caffe
