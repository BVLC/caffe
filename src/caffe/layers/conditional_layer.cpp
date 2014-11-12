#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConditionalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  conditional_index_ = this->layer_param_.conditional_param().conditional_index();
  CHECK_LE(conditional_index_,bottom[0]->channels()-1 ) <<
    "conditional_index_ should be <= bottom[0]->channels()-1";
}

template <typename Dtype>
void ConditionalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //bottom[0] is the "IF_blob"
  //bottom[1] is the "THEN_blob"
  //top[0] is the vector of indices that passed the conditional check
  //top[1] is the top to which will be forwarded bottom[1]
 
 // top[0]->Reshape(1, 1, 1, 1);
 // top[1]->Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
  
  
  const Dtype* bottom_data_IF = bottom[0]->cpu_data();
  const Dtype* bottom_data_THEN = bottom[1]->cpu_data();
  
  int num_items = bottom[0]->num();
  int num_channels = bottom[0]->channels();
  indices_to_keep_.clear();
  //look through the batch to find who passes the conditional check
  for (size_t item_id = 0; item_id < num_items; ++item_id) {
    
    int index_IF = item_id*num_channels;
    const Dtype* tmp_data_IF = bottom_data_IF + index_IF;
    int argmax = std::distance(tmp_data_IF,
        std::max_element(tmp_data_IF, tmp_data_IF + num_channels));
    
    if (argmax == conditional_index_)
      indices_to_keep_.push_back(item_id);
      
  }
  
  //Only items that passed conditional check will be forwarded
  int new_tops_num = indices_to_keep_.size();
  if (new_tops_num == 0)
    new_tops_num = 1;

  top[0]->Reshape(new_tops_num, 1, 1, 1);
  top[1]->Reshape(new_tops_num, bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());


  

  
  
}

template <typename Dtype>
void ConditionalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data_IF = bottom[0]->cpu_data();
  const Dtype* bottom_data_THEN = bottom[1]->cpu_data();
  
  Dtype* top_data_indices = top[0]->mutable_cpu_data();
  Dtype* top_data = top[1]->mutable_cpu_data();
  
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
void ConditionalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(ConditionalLayer);
#endif

INSTANTIATE_CLASS(ConditionalLayer);
REGISTER_LAYER_CLASS(CONDITIONAL, ConditionalLayer);
}  // namespace caffe
