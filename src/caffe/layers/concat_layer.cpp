// Copyright 2014 Sergio Guadarrama

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConcatLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_GT(bottom.size(), 1) << "Concat Layer takes at least two blobs as input.";
  CHECK_EQ(top->size(), 1) << "Concat Layer takes a single blob as output.";
  concat_dim_ = this->layer_param_.concat_dim();
  CHECK_GE(concat_dim_,0) << "concat_dim should be >= 0";
  CHECK_LE(concat_dim_,1) << 
    "For now concat_dim <=1, it can only concat num and channels";
  // Intialize with the first blob
  COUNT_ = bottom[0]->count();
  NUM_ = bottom[0]->num();
  CHANNELS_ = bottom[0]->channels();
  HEIGHT_ = bottom[0]->height();
  WIDTH_ = bottom[0]->width();  
  for (int i=1; i<bottom.size(), ++i) {
    COUNT_ += bottom[i]->count();
    if (concat_dim_==0) {
      NUM_ += bottom[i]->num();  
    } else if (concat_dim_ == 1){
      CHANNELS_ += bottom[i]->channels();
    } else if (concat_dim_ == 2) {
      HEIGHT_ += bottom[i]->height();
    } else if (concat_dim_ == 3) {
      WIDTH_ += bottom[i]->width(); 
    }
  }
  (*top)[0]->Reshape(NUM_, CHANNELS_, HEIGHT_, WIDTH_); 
  CHECK_EQ(COUNT_, (*top)[0]->count());
};

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {  
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  if (concat_dim_==0) {
    int offset_num = 0;    
    for (i=0; i<bottom.size(); ++i) {      
      Dtype* bottom_data = bottom[i]->cpu_data();
      caffe_copy(bottom[i]->count(), bottom_data, top_data[(*top)[0]->offset(offset_num));
      offset_num += bottom[i]->num();
    }
  } else if (concat_dim_ == 1) {
    int offset_channel = 0;    
    for (i=0; i<bottom.size(); ++i) {
      Dtype* bottom_data = bottom[i]->cpu_data();
      int num_elem = bottom[i]->channels()*bottom[i]->height()*bottom[i]->width();
      for (n=0; n<NUM_; ++n){
        caffe_copy(num_elem, bottom_data[bottom[i]->offset(n),
          top_data[(*top)[0]->offset(n,offset_channel)]);  
      }          
      offset_channel += bottom[i]->channels();
    }
  } else {
    LOG(FATAL) << "concat_dim along dim" << concat_dim << " not implemented yet";
  }  
}

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  if (concat_dim_==0) {
    int offset_num = 0;    
    for (i=0; i<bottom.size(); ++i) {      
      Dtype* bottom_data = bottom[i]->gpu_data();
      caffe_gpu_copy(bottom[i]->count(), bottom_data, top_data[(*top)[0]->offset(offset_num));
      offset_num += bottom[i]->num();
    }
  } else if (concat_dim_ == 1) {
    int offset_channel = 0;    
    for (i=0; i<bottom.size(); ++i) {
      Dtype* bottom_data = bottom[i]->gpu_data();
      int num_elem = bottom[i]->channels()*bottom[i]->height()*bottom[i]->width();
      for (n=0; n<NUM_; ++n){
        caffe_gpu_copy(num_elem, bottom_data[bottom[i]->offset(n),
          top_data[(*top)[0]->offset(n,offset_channel)]);  
      }          
      offset_channel += bottom[i]->channels();
    }
  } else {
    LOG(FATAL) << "concat_dim along dim" << concat_dim << " not implemented yet";
  }  
}

template <typename Dtype>
Dtype ConcatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff(); 
  if (concat_dim_==0) {
    int offset_num = 0;    
    for (i=0; i < bottom->size(); ++i) {
      Blob<Dtype>* blob = (*bottom)[i];
      Dtype* bottom_diff = blob->mutable_cpu_diff();
      caffe_copy(blob->count(), top_diff[top[0]->offset(offset_num)],bottom_diff);
      offset_num += blob->num();
    }
  } else if (concat_dim_ == 1) {
    int offset_channel = 0;    
    for (i=0; i < bottom->size(); ++i) {
      Blob<Dtype>* blob = (*bottom)[i];
      Dtype* bottom_diff = blob->mutable_cpu_diff();
      int num_elem = blob->channels()*blob->height()*blob->width();
      for (n=0; n<NUM_; ++n){
        caffe_copy(num_elem, top_diff[top[0]->offset(n,offset_channel)], 
          bottom_diff[blob->offset(n)]);  
      }          
      offset_channel += blob->channels();
    }
  } else {
    LOG(FATAL) << "concat_dim along dim" << concat_dim << " not implemented yet";
  } 
  return Dtype(0.);
}


template <typename Dtype>
Dtype ConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff(); 
  if (concat_dim_==0) {
    int offset_num = 0;    
    for (i=0; i < bottom->size(); ++i) {
      Blob<Dtype>* blob = (*bottom)[i];
      Dtype* bottom_diff = blob->mutable_gpu_diff();
      caffe_gpu_copy(blob->count(), top_diff[top[0]->offset(offset_num)],bottom_diff);
      offset_num += blob->num();
    }
  } else if (concat_dim_ == 1) {
    int offset_channel = 0;    
    for (i=0; i < bottom->size(); ++i) {
      Blob<Dtype>* blob = (*bottom)[i];
      Dtype* bottom_diff = blob->mutable_gpu_diff();
      int num_elem = blob->channels()*blob->height()*blob->width();
      for (n=0; n<NUM_; ++n){
        caffe_gpu_copy(num_elem, top_diff[top[0]->offset(n,offset_channel)], 
          bottom_diff[blob->offset(n)]);  
      }          
      offset_channel += blob->channels();
    }
  } else {
    LOG(FATAL) << "concat_dim along dim" << concat_dim << " not implemented yet";
  } 
  return Dtype(0.);
}

INSTANTIATE_CLASS(ConcatLayer);

}  // namespace caffe
