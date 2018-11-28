#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_saliency_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_output_shape() {
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
void ConvolutionSaliencyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
  this->compute_output_shape();
  if (this->mask_term_) {
    if (this->layer_param_.convolution_saliency_param().saliency()==2) {
        output_saliencies_channel_.Reshape({this->max_saliency, top[0]->shape()[1]});
    }
    else {
        output_saliencies_channel_.Reshape({1, top[0]->shape()[1]});
    }
    output_saliencies_points_.Reshape(top[0]->shape()); //shape nchw
    output_saliencies_filter_.Reshape({top[0]->shape()[0], top[0]->shape()[1]}); //shape nc
  }
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }

    Dtype* channel_saliency_data = output_saliencies_channel_.mutable_cpu_data();    
  
    switch (this->layer_param_.convolution_saliency_param().saliency()) {
      case (0): { // Fisher Information
        compute_fisher_cpu(bottom_data, bottom_diff, channel_saliency_data);
      } break;

      case (1): { // Taylor Series
        compute_taylor_cpu(bottom_data, bottom_diff, channel_saliency_data);
      } break;

      case (2): {
        compute_fisher_cpu(bottom_data, bottom_diff, channel_saliency_data);
        compute_taylor_cpu(bottom_data, bottom_diff, channel_saliency_data + this->output_shape_[1]);
      } break;

      default: {
      } break;
    }
  }
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_fisher_cpu(const Dtype *  act_data, const Dtype *  act_diff, Dtype * fisher_info) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_cpu_data();    
  Dtype* filter_saliency_data = output_saliencies_filter_.mutable_cpu_data();    
  
  caffe_mul(this->output_saliencies_points_.count(0,3), act_data, act_diff, output_saliency_data);
  
  for (int i = 0; i < this->output_saliencies_points_.count(0, 1); ++i) {
    caffe_sum(output_saliencies_points_.count(2,3), output_saliency_data, filter_saliency_data);
    output_saliency_data += output_saliencies_points_.count(2,3);
    ++filter_saliency_data;
  }
  filter_saliency_data = output_saliencies_filter_.mutable_cpu_data();    
  
  caffe_powx(this->output_saliencies_filter_.count(0, 1), filter_saliency_data, (Dtype)2, filter_saliency_data);
  
  Dtype * original_channel = fisher_info;
  for (int i = 0; i < this->output_shape_[1]; ++i ) { 
    *fisher_info = caffe_cpu_asum(this->output_shape_[0], filter_saliency_data, this->output_shape_[1]); // functionally it does not matter if we use sum or asum; sum across batches
    filter_saliency_data += 1;
    ++fisher_info;
  }
  fisher_info = original_channel;
  
  caffe_scal(this->output_shape_[1], 1/(Dtype)(this->output_shape_[0]*2), fisher_info);
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_cpu(const Dtype *  act_data, const Dtype *  act_diff, Dtype * taylor) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_cpu_data();    
  Dtype* filter_saliency_data = output_saliencies_filter_.mutable_cpu_data();    
  
  caffe_mul(this->output_saliencies_points_.count(0,3), act_data, act_diff, output_saliency_data);
  caffe_abs(this->output_saliencies_points_.count(0,3), output_saliency_data, output_saliency_data);
  
  for (int i = 0; i < this->output_saliencies_points_.count(0, 1); ++i) {
    *filter_saliency_data = caffe_cpu_asum(output_saliencies_points_.count(2,3), output_saliency_data);
    output_saliency_data += output_saliencies_points_.count(2,3);
    ++filter_saliency_data;
  }
  filter_saliency_data = output_saliencies_filter_.mutable_cpu_data();    
  
  Dtype * original_channel = taylor;
  for (int i = 0; i < this->output_shape_[1]; ++i ) { 
    *taylor = caffe_cpu_asum(this->output_shape_[0], filter_saliency_data, this->output_shape_[1]); // functionally it does not matter if we use sum or asum; sum across batches
    filter_saliency_data += 1;
    ++taylor;
  }
  taylor = original_channel;
  
  caffe_scal(this->output_shape_[1], 1/(Dtype)(this->output_shape_[0]*this->output_shape_[2]*this->output_shape_[3]), taylor);
  
}


#ifdef CPU_ONLY
STUB_GPU(ConvolutionSaliencyLayer);
#endif

INSTANTIATE_CLASS(ConvolutionSaliencyLayer);

}  // namespace caffe
