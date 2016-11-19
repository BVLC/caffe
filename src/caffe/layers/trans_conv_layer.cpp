#include <vector>

#include "caffe/layers/trans_conv_layer.hpp"

namespace caffe {


template <typename Dtype>
void TransformerConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
  
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + this->channel_axis_);
  top_shape.push_back(this->num_output_);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    top_shape.push_back(this->output_shape_[i]);
  }
  top_shape[this->channel_axis_] *= 8;
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
}


template <typename Dtype>
void TransformerConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}


template <typename Dtype>
void TransformerConvolutionLayer<Dtype>::get_weight_diff(vector<Dtype*> weight_diffs, 
    Dtype* weight_diff, TransformerConvParameter param){
  if (param.action() == 0){ // rotation 8 kernels
    // only used for 3x3 kernel
    int circle[8] = {0, 1, 2, 5, 8, 7, 6, 3};
    for (int i = 0; i < 8; ++i){
      weight_diff[4] += weight_diffs[i][4];
      for (int j = 0; j < 8; ++j){
        weight_diff[circle[i]] += weight_diffs[j][circle[(i+j)%8]];
      }
    }
  }else if (param.action() == 1){ // flip 3 kernels
    // not implemented
  }
}


template <typename Dtype>
vector<Dtype*> TransformerConvolutionLayer<Dtype>::get_trans_weights(const Dtype* weight,
      TransformerConvParameter param){
  LOG(INFO) << "======get_trans_weights=1=======";
  Dtype* input =  new Dtype[9 * this->channels_ * this->num_output_];
  LOG(INFO) << "======get_trans_weights=2=======";
  for (int i = 0; i < 9 * this->channels_ * this->num_output_; ++i){
    input[i] = weight[i];
  }
  vector<Dtype*> weights(8);
  weights[0] = input;
  if (param.action() == 0){ // rotation 8 kernels
    // only used for 3x3 kernel
    int circle[8] = {0, 1, 2, 5, 8, 7, 6, 3};
    for (int step = 1; step < 8; ++step){
      Dtype* curWeight = new Dtype[9 * this->channels_ * this->num_output_];
      for (int i = 0; i < this->channels_*this->num_output_; ++i){
        curWeight[i*9+4] = input[i*9+4];
        for (int j = 0; j < 8; ++j){
          int new_idnex = circle[(j+step)%8];
          curWeight[i*9+new_idnex] = input[i*9+circle[j]];
        }
      }
      weights[step] = curWeight;
    }
  }else if (param.action() == 1){ // flip 3 kernels
    // not implemented
  }
  return weights;
}


template <typename Dtype>
void TransformerConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  TransformerConvParameter param = this->layer_param_.trans_conv_param();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  vector<Dtype*> weights = get_trans_weights(weight, param);
  int weight_size = weights.size();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      for (int j = 0; j < weight_size; ++j){
        this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weights[j],
            top_data + (n*weight_size+j) * this->top_dim_);
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->cpu_data();
          this->forward_cpu_bias(top_data + (n*weight_size+j) * this->top_dim_, bias);
        } 
      }
    }
  }
  for (int i = 0; i < weight_size; ++i){
    delete[] weights[i];
  }
}

template <typename Dtype>
void TransformerConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  TransformerConvParameter param = this->layer_param_.trans_conv_param();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  vector<Dtype*> weights = get_trans_weights(weight, param);
  vector<Dtype*> weight_diffs = get_trans_weights(weight_diff, param);
  int weight_size = weights.size();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    for (int k = 0; k < this->bottom_dim_; ++k){
      LOG(INFO) << "===bottom diff init: "<< " -- " << bottom_diff[k];
    }
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        for (int j = 0; j < weight_size; ++j){
          this->backward_cpu_bias(bias_diff, top_diff + (n*weight_size+j) * this->top_dim_);
        }
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        for (int j = 0; j < weight_size; ++j){
          Dtype bottom_diff_temp[this->bottom_dim_];
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          if (this->param_propagate_down_[0]) {
            this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                top_diff + (n*weight_size+j) * this->top_dim_, weight_diffs[j]);
          }
          // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i]) {
            this->backward_cpu_gemm(top_diff + (n*weight_size+j) * this->top_dim_, weights[j],
                bottom_diff_temp);
            caffe_add(this->bottom_dim_, bottom_diff + n * this->bottom_dim_, bottom_diff_temp,
                bottom_diff + n * this->bottom_dim_);
          }
        }
      }
    }
    get_weight_diff(weight_diffs, weight_diff, param);
  }
  for (int i = 0; i < weight_size; ++i){
    delete[] weights[i];
    delete[] weight_diffs[i];
  }
}

#ifdef CPU_ONLY
STUB_GPU(TransformerConvolutionLayer);
#endif

INSTANTIATE_CLASS(TransformerConvolutionLayer);
REGISTER_LAYER_CLASS(TransformerConvolution);

}  // namespace caffe
