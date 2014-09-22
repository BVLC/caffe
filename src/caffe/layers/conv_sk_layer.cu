#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionSKLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = (*top)[i]->mutable_gpu_data();
    Dtype* col_data = col_buffer_.mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    int weight_offset = M_ * K_;
    int col_offset = K_ * N_;
    int top_offset = M_ * N_;
    
    for (int n = 0; n < num_; ++n) {
       // First, im2col
       im2col_sk_gpu(bottom_data + bottom[i]->offset(n), channels_, height_,
           width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
           kstride_h_, kstride_w_,
           col_data);
       // Second, innerproduct with groups
       for (int g = 0; g < group_; ++g) {
         caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
           (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
           (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g);
       } 
       // third, add bias
       if (bias_term_) {
         caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
            N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
            bias_multiplier_.gpu_data(),
            (Dtype)1., top_data + (*top)[i]->offset(n));
       }
    }
  }
  return;
}

template <typename Dtype>
void ConvolutionSKLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
   LOG(FATAL) << "Backward_gpu() not implemented for ConvolutionSKLayer";  
}


INSTANTIATE_CLASS(ConvolutionSKLayer);

}  // namespace caffe
