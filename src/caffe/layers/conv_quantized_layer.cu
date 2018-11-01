#include <vector>

#include "caffe/layers/conv_quantized_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionQuantizedLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_quantized = this->weights_quantized_.mutable_gpu_data();
  const int count = this->blobs_[0]->count();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    switch (this->layer_param_.convolution_quantized_param().method()) {
      case (0): {
        unsigned no_centroids = this->layer_param_.convolution_quantized_param().centroids();
        const Dtype* centroids = this->centroids_.gpu_data();
        //TODO: call kmeans here! results in weight_quantized, and delete the line below
        caffe_copy(count, weight, weight_quantized);
      } break;

      case(1): {
        std::bitset<8*sizeof(Dtype)> mantissa_mask;
        mantissa_mask.flip();
        mantissa_mask <<= this->layer_param_.convolution_quantized_param().truncate_bits();
        caffe_gpu_and(count, mantissa_mask, weight, weight_quantized);
      } break;

      default: {
        caffe_copy(count, weight, weight_quantized);
      } break;
    }
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight_quantized,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionQuantizedLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
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
      // Compute saliency for each output
      int outputs = 1;
      for (int i = 0; i < this->output_shape_.size(); i++) {
        outputs *= this->output_shape_[i];
      }

      switch (this->layer_param_.convolution_quantized_param().saliency()) {
        case (0): { // Fisher Information
          Dtype* saliency_data = this->output_saliencies_.mutable_gpu_data();
          caffe_gpu_mul(outputs, bottom_data, bottom_diff, saliency_data);
          caffe_gpu_powx(outputs, saliency_data, (Dtype)2, saliency_data);

          Dtype* centroids = this->centroids_.mutable_cpu_data();
          // TODO: update the centroids here with the saliency data
        } break;

        case (1): { // Taylor Series
          Dtype* saliency_data = this->output_saliencies_.mutable_gpu_data();
          caffe_copy(outputs, bottom_data, saliency_data);

          Dtype* centroids = this->centroids_.mutable_gpu_data();
          // TODO: update the centroids here with the saliency data
        } break;

        case (2): { // Magnitude
          Dtype* saliency_data = this->output_saliencies_.mutable_gpu_data();
          caffe_copy(outputs, bottom_data, saliency_data);

          Dtype* centroids = this->centroids_.mutable_gpu_data();
          // TODO: update the centroids here with the saliency data
        } break;

        default: {
          Dtype* saliency_data = this->output_saliencies_.mutable_gpu_data();
          caffe_copy(outputs, bottom_data, saliency_data);

          Dtype* centroids = this->centroids_.mutable_gpu_data();
          // TODO: update the centroids here with the saliency data
        } break;
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionQuantizedLayer);

}  // namespace caffe
