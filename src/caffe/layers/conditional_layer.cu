#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConditionalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data_IF = bottom[0]->gpu_data();
  const Dtype* bottom_data_THEN = bottom[1]->gpu_data();
  
  Dtype* top_data_indices = top[0]->mutable_gpu_data();
  Dtype* top_data = top[1]->mutable_gpu_data();
  
  int new_tops_num = indices_to_keep_.size();
  
  caffe_copy(new_tops_num, &indices_to_keep_[0],
        top_data_indices);
  
  size_t size_top = top[1]->count()/top[1]->num();
  for (size_t n = 0; n<new_tops_num; n++)
  {
    int offset = indices_to_keep_[n];    
    int data_offset_top = size_top*n;
    int data_offset_bottom = size_top*offset;

    caffe_copy(size_top, bottom_data_THEN+data_offset_bottom,
        top_data+data_offset_top);
  }


}

template <typename Dtype>
void ConditionalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(ConditionalLayer);
}  // namespace caffe
