#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/trans_conv_layer.hpp"

namespace caffe {


template <typename Dtype>
void TransformerConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  this->num_output_ = this->layer_param_.convolution_param().num_output();
  this->num_output_ /= 8;
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  // ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  // int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
  // if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
  //   kernel_shape_data[0] = conv_param.kernel_h();
  //   kernel_shape_data[1] = conv_param.kernel_w();
  // } else {
  //   const int num_kernel_dims = conv_param.kernel_size_size();
  //   for (int i = 0; i < this->num_spatial_axes_; ++i) {
  //     kernel_shape_data[i] =
  //         conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
  //   }
  // }
  // // Configure output channels and groups.
  // this->channels_ = bottom[0]->shape(this->channel_axis_);
  // this->num_output_ = this->layer_param_.convolution_param().num_output();
  // this->num_output_ /= 8;
  // this->group_ = this->layer_param_.convolution_param().group();
  // // Handle the parameters: weights and biases.
  // // - blobs_[0] holds the filter weights
  // // - blobs_[1] holds the biases (optional)
  // vector<int> weight_shape(2);
  // weight_shape[0] = this->num_output_;
  // weight_shape[1] = this->channels_ / this->group_;
  // for (int i = 0; i < this->num_spatial_axes_; ++i) {
  //   weight_shape.push_back(kernel_shape_data[i]);
  // }
  // this->bias_term_ = this->layer_param_.convolution_param().bias_term();
  // vector<int> bias_shape(this->bias_term_, this->num_output_);
  // // Initialize and fill the weights:
  // // output channels x input channels per-group x kernel height x kernel width
  // this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
  // shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
  //     this->layer_param_.convolution_param().weight_filler()));
  // weight_filler->Fill(this->blobs_[0].get());
  // // If necessary, initialize and fill the biases.
  // if (this->bias_term_) {
  //   this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
  //   shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
  //       this->layer_param_.convolution_param().bias_filler()));
  //   bias_filler->Fill(this->blobs_[1].get());
  // }
  // this->weight_offset_ = this->num_output_ * this->blobs_[0]->count(1) / this->group_;
  // // Propagate gradients to the parameters (as directed by backward pass).
  // this->param_propagate_down_.resize(this->blobs_.size(), true);
}


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
void TransformerConvolutionLayer<Dtype>::get_weight_diff(Dtype* weight_diffs, 
    Dtype* weight_diff, TransformerConvParameter param){
  int count = 9 * this->channels_ * this->num_output_;
  Dtype diff_temp[count];
  caffe_set(count, (Dtype) 0.0, diff_temp);
  if (param.action() == 0){ // rotation 8 kernels
    // only used for 3x3 kernel
    int circle[8] = {0, 1, 2, 5, 8, 7, 6, 3};
    for (int i = 0; i < 8; ++i){
      for (int n = 0; n < this->channels_ * this->num_output_; ++n){
        diff_temp[n*9+4] += weight_diffs[i*count+n*9+4];
        for (int j = 0; j < 8; ++j){
          diff_temp[n*9+circle[i]] += weight_diffs[i*count+n*9+circle[(i+j)%8]];
        }
      }
    }
  }else if (param.action() == 1){ // flip 3 kernels
    // not implemented
  }
  caffe_copy(count, diff_temp, weight_diff);
}


template <typename Dtype>
void TransformerConvolutionLayer<Dtype>::get_trans_weights(Dtype* weights, const Dtype* weight,
      TransformerConvParameter param){
  int count = 9 * this->channels_ * this->num_output_;
  int new_index;
  caffe_copy(count, weight, weights);
  if (param.action() == 0){ // rotation 8 kernels
    // only used for 3x3 kernel
    int circle[8] = {0, 1, 2, 5, 8, 7, 6, 3};
    Dtype curWeight[count];
    for (int step = 1; step < 8; ++step){
      for (int i = 0; i < this->channels_*this->num_output_; ++i){
        caffe_copy(1, weight+i*9+4, curWeight+i*9+4);
        for (int j = 0; j < 8; ++j){
          new_index = circle[(j+step)%8];
          caffe_copy(1, weight+i*9+circle[j], curWeight+i*9+new_index);
        }
      }
      caffe_copy(count, curWeight, weights+step*count);
    }
  }else if (param.action() == 1){ // flip 3 kernels
    // not implemented
  }
}


template <typename Dtype>
void TransformerConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int count = 9 * this->channels_ * this->num_output_;
  TransformerConvParameter param = this->layer_param_.trans_conv_param();
  Dtype* weight = this->blobs_[0]->mutable_cpu_data();
  Dtype weights[8 * count];
  get_trans_weights(weights, weight, param);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      for (int j = 0; j < 8; ++j){
        caffe_copy(count, weights+j*count, weight);
        this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + (n*8+j) * this->top_dim_);
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->cpu_data();
          this->forward_cpu_bias(top_data + (n*8+j) * this->top_dim_, bias);
        } 
      }
    }
  }
  caffe_copy(count, weights, weight);
}

template <typename Dtype>
void TransformerConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(INFO) << "Backward_cpu===1==>";
  int count = 9 * this->channels_ * this->num_output_;
  TransformerConvParameter param = this->layer_param_.trans_conv_param();
  Dtype* weight = this->blobs_[0]->mutable_cpu_data();
  for (int i = 0; i < count; ++i){
    LOG(INFO) << "Backward_cpu===2==>" << weight[i]; 
  }

  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype weights[8 * count];
  Dtype weight_diffs[8 * count];
  Dtype diff_temp[count];
  Dtype bottom_temp[this->bottom_dim_];
  get_trans_weights(weights, weight, param);
  get_trans_weights(weight_diffs, weight_diff, param);
  Dtype bottom_diff_temp[this->bottom_dim_];
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        for (int j = 0; j < 8; ++j){
          this->backward_cpu_bias(bias_diff, top_diff + (n*8+j) * this->top_dim_);
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
            this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                top_diff + (n*8+j) * this->top_dim_, weight_diff);
            caffe_copy(count, weight_diff, diff_temp);
            caffe_add(count, diff_temp, weight_diffs+j*count, weight_diffs+j*count);
          }
          // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i]) {
            caffe_copy(count, weights+j*count, weight);
            this->backward_cpu_gemm(top_diff + (n*8+j) * this->top_dim_, weight,
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
  LOG(INFO) << "Backward_cpu===10==>";
  get_weight_diff(weight_diffs, weight_diff, param);
  LOG(INFO) << "Backward_cpu===ok==>";
}

#ifdef CPU_ONLY
STUB_GPU(TransformerConvolutionLayer);
#endif

INSTANTIATE_CLASS(TransformerConvolutionLayer);
REGISTER_LAYER_CLASS(TransformerConvolution);

}  // namespace caffe
