#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_saliency_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
    // Compute saliency for each output
    int outputs = 1;
    for (int i = 0; i < this->output_shape_.size(); i++) {
      outputs *= this->output_shape_[i];
    }
    int filters = 1;
    for (int i = 2; i < this->output_shape_.size(); i++) {
      filters *= this->output_shape_[i];
    }

    Dtype* output_saliency_data = output_saliencies_points_.mutable_gpu_data();    
    Dtype* filter_saliency_data = output_saliencies_filter_.mutable_gpu_data();    
    Dtype* channel_saliency_data = output_saliencies_filter_.mutable_gpu_data();    
  
    switch (this->layer_param_.convolution_saliency_param().saliency()) {
      case (0): { // Fisher Information
        caffe_gpu_mul(outputs, bottom_data, bottom_diff, output_saliency_data);
        for (int i = 0; i < output_saliencies_points_.count(0, 1); ++i) {
          caffe_gpu_sum(output_saliencies_points_.count(1), output_saliency_data, filter_saliency_data);
          output_saliency_data += output_saliencies_channel_.count(1);
          ++filter_saliency_data;
        }
        filter_saliency_data = output_saliencies_filter_.mutable_gpu_data();    
        caffe_powx(filters, filter_saliency_data, (Dtype)2, filter_saliency_data);
        for (int i = 0; i < this->output_shape_[1]; ++i ) { 
          caffe_gpu_asum(this->output_shape_[0], filter_saliency_data, channel_saliency_data, this->output_shape_[1]); // functionally it does not matter if we use sum or asum; sum across batches
          filter_saliency_data += 1;
          ++channel_saliency_data;
        }
        channel_saliency_data = output_saliencies_filter_.mutable_gpu_data();
        caffe_scal(this->output_shape_[1], 1/(Dtype)(this->output_shape_[0]*this->output_shape_[2]*this->output_shape_[3]), channel_saliency_data);
      } break;

      case (1): { // Taylor Series
        //caffe_gpu_mul(outputs, bottom_data, bottom_diff, output_saliency_data);
      } break;

      default: {
      } break;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionSaliencyLayer);

}  // namespace caffe
