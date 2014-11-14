#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConditionalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data_THEN = bottom[1]->gpu_data();
  const Dtype* bottom_data_LABELS = bottom[2]->gpu_data();
  
  
  Dtype* top_data_indices_OR_labels = top[0]->mutable_gpu_data();
  Dtype* top_data = top[1]->mutable_gpu_data();
  
  int new_tops_num = indices_to_keep_.size();
  
  if(output_type_ == ConditionalParameter_OUTPUT_TYPE_FILTERED_INDICES) {
    caffe_copy(new_tops_num, &indices_to_keep_[0],
          top_data_indices_OR_labels);
  }
  
  size_t size_single_batch = top[1]->count()/top[1]->num();
  size_t size_single_label = bottom[2]->count()/bottom[2]->num();
  for (size_t n = 0; n<new_tops_num; n++)
  {
    int offset = indices_to_keep_[n];    
    int data_offset_top = size_single_batch*n;
    int data_offset_bottom = size_single_batch*offset;

    caffe_copy(size_single_batch, bottom_data_THEN+data_offset_bottom,
        top_data+data_offset_top);
        
    if(output_type_ == ConditionalParameter_OUTPUT_TYPE_FILTERED_LABELS) {    
      int data_offset_top_labels = size_single_label*n;
      int data_offset_bottom_labels = size_single_label*offset;

      caffe_copy(size_single_label, bottom_data_LABELS+data_offset_bottom_labels,
          top_data_indices_OR_labels+data_offset_top_labels);
    }
  }
}

template <typename Dtype>
void ConditionalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) { return; }

    int size_single_batch = top[1]->count()/top[1]->num();
    int index_top = 0;
    std::vector<double> zeros (size_single_batch,0.0);
    for (size_t n = 0; n<bottom[1]->num(); n++)
    {
      int offset = indices_to_keep_[n];
      int data_offset_bottom = size_single_batch*n;
      if(n != offset) { //this data was not been forwarded
        caffe_copy(size_single_batch,(Dtype*)&zeros[0], bottom[1]->mutable_gpu_diff() + data_offset_bottom);
      }
      else { //this data was been forwarded
        int data_offset_top = size_single_batch*index_top;
        index_top++;
        caffe_copy(size_single_batch,  top[1]->mutable_gpu_diff() + data_offset_top, bottom[1]->mutable_gpu_diff() + data_offset_bottom);
      }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConditionalLayer);
}  // namespace caffe
