#include <vector>

#include "caffe/layers/trans_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void TransForward(TransformerConvolutionLayer<Dtype>* this, 
      const int nthreads, Dtype* weights, Dtype* weight,
      const Dtype* bottom_data, Dtype* top_data, const Dtype* bias){
  CUDA_KERNEL_LOOP(index, nthreads) {
    n = index / 8;
    j = index % 8;
    caffe_copy(count, weights+j*count, weight);
    this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
        top_data + (n*8+j) * this->top_dim_);
    if (this->bias_term_) {
      this->forward_gpu_bias(top_data + (n*8+j) * this->top_dim_, bias);
    }
  }
}

template <typename Dtype>
void TransformerConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int count = 9 * this->channels_ * this->num_output_;
  TransformerConvParameter param = this->layer_param_.trans_conv_param();
  Dtype* weight = this->blobs_[0]->mutable_gpu_data();
  Dtype weights[8 * count];
  int nthreads = this->num_ * 8;
  int n, j;
  get_trans_weights(weights, weight, param);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    const Dtype* bias = this->blobs_[1]->gpu_data();
    TransForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      this, nthreads, weights, weight, bottom_data, top_data);
  }
  caffe_copy(count, weights, weight);
}


__global__ void TransBackwardBias(TransformerConvolutionLayer<Dtype>* this,
    const int nthreads, Dtype* bias_diff, Dtype* top_diff){
  CUDA_KERNEL_LOOP(index, nthreads) {
    n = index / 8;
    j = index % 8;
    this->backward_gpu_bias(bias_diff, top_diff + (n*8+j) * this->top_dim_);
  }
}

__global__ void TransBackward(TransformerConvolutionLayer<Dtype>* this,
      const int nthreads, int count, Dtype* weights, Dtype* weight,
      const Dtype* bottom_data, Dtype* weight_diff, Dtype* weight_diffs, Dtype* top_diff, 
      Dtype* bottom_diff, const bool propagate_down){
  int pre_n = -1;
  Dtype diff_temp[count];
  Dtype bottom_temp[this->bottom_dim_];
  Dtype bottom_diff_temp[this->bottom_dim_];
  CUDA_KERNEL_LOOP(index, nthreads) {
    n = index / 8;
    j = index % 8;
    if (pre_n != n){
      caffe_set(this->bottom_dim_, (Dtype)0.0, bottom_diff_temp);
    }
    if (this->param_propagate_down_[0]) {
      caffe_set(count, (Dtype) 0.0, diff_temp);
      caffe_copy(count, diff_temp, weight_diff);
      this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
          top_diff + (n*8+j) * this->top_dim_, weight_diff);
      caffe_copy(count, weight_diff, diff_temp);
      caffe_add(count, diff_temp, weight_diffs+j*count, weight_diffs+j*count);
    }
    // gradient w.r.t. bottom data, if necessary.
    if (propagate_down) {
      caffe_copy(count, weights+j*count, weight);
      this->backward_gpu_gemm(top_diff + (n*8+j) * this->top_dim_, weight,
          bottom_diff + n * this->bottom_dim_);
      caffe_copy(this->bottom_dim_, bottom_diff + n * this->bottom_dim_, bottom_temp);
      caffe_add(this->bottom_dim_, bottom_temp, bottom_diff_temp, bottom_diff_temp);
    }
    if (pre_n != n && pre_n >= 0){
      caffe_copy(this->bottom_dim_, bottom_diff_temp, bottom_diff + n * this->bottom_dim_);
      pre_n = n;
    }
  }
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
  get_trans_weights(weights, weight, param);
  get_trans_weights(weight_diffs, weight_diff, param);
  int nthreads = this->num_ * 8;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      TransBackwardBias<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          this, nthreads, bias_diff, top_diff);
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      TransBackward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          this, nthreads, count, Dtype* weights, Dtype* weight, bottom_data, weight_diff, 
          weight_diffs, top_diff, bottom_diff, propagate_down[i]);
  }
  caffe_copy(count, weights, weight);
  get_weight_diff(weight_diffs, weight_diff, param);
}

INSTANTIATE_LAYER_GPU_FUNCS(TransformerConvolutionLayer);

}  // namespace caffe
