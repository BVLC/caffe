#include <vector>

#include "caffe/layers/trans_conv_layer.hpp"

namespace caffe {


template <typename Dtype>
void TransformerConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int count = 9 * this->channels_ * this->num_output_;
  TransformerConvParameter param = this->layer_param_.trans_conv_param();
  Dtype* weight = this->blobs_[0]->mutable_gpu_data();
  Dtype weights[8 * count];
  get_trans_weights(weights, weight, param);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      for (int j = 0; j < 8; ++j){
        //caffe_gpu_memcpy(count, weights+j*count, (Dtype*) curWeight);
        caffe_copy(count, weights+j*count, weight);
        this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + (n*8+j) * this->top_dim_);
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->gpu_data();
          this->forward_gpu_bias(top_data + (n*8+j) * this->top_dim_, bias);
        } 
      }
    }
  }
  caffe_copy(count, weights, weight);
}


template <typename Dtype>
void TransformerConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = 9 * this->channels_ * this->num_output_;
  TransformerConvParameter param = this->layer_param_.trans_conv_param();
  Dtype* weight = this->blobs_[0]->mutable_gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype weights[8 * count];
  Dtype weight_diffs[8 * count];
  Dtype diff_temp[count];
  Dtype bottom_temp[this->bottom_dim_];
  Dtype bottom_diff_temp[this->bottom_dim_];
  get_trans_weights(weights, weight, param);
  get_trans_weights(weight_diffs, weight_diff, param);
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        for (int j = 0; j < 8; ++j){
          this->backward_gpu_bias(bias_diff, top_diff + (n*8+j) * this->top_dim_);
        }
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        caffe_set(this->bottom_dim_, (Dtype)0.0, bottom_diff_temp);
        for (int j = 0; j < 8; ++j){
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          if (this->param_propagate_down_[0]) {
            caffe_set(count, (Dtype) 0.0, diff_temp);
            caffe_copy(count, diff_temp, weight_diff);
            this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
                top_diff + (n*8+j) * this->top_dim_, weight_diff);
            caffe_copy(count, weight_diff, diff_temp);
            caffe_add(count, diff_temp, weight_diffs+j*count, weight_diffs+j*count);
          }
          // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i]) {
            caffe_copy(count, weights+j*count, weight);
            this->backward_gpu_gemm(top_diff + (n*8+j) * this->top_dim_, weight,
                bottom_diff + n * this->bottom_dim_);
            caffe_copy(this->bottom_dim_, bottom_diff + n * this->bottom_dim_, bottom_temp);
            caffe_add(this->bottom_dim_, bottom_temp, bottom_diff_temp, bottom_diff_temp);
          }
        }
        caffe_copy(this->bottom_dim_, bottom_diff_temp, bottom_diff + n * this->bottom_dim_);
      }
    }
  }
  caffe_copy(count, weights, weight);
  get_weight_diff(weight_diffs, weight_diff, param);
}

INSTANTIATE_LAYER_GPU_FUNCS(TransformerConvolutionLayer);

}  // namespace caffe
