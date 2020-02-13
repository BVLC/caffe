#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void spatial_conv_fwd_kernel(const int count, const int channels, const int output_height, const int output_width,
            const int input_height, const int input_width, 
            const int kernel_height, const int kernel_width,
            const int stride_y, const int stride_x, const int pad_y, const int pad_x,
            const Dtype *input_data, const Dtype *weight_data, Dtype *output_data) { //, const Dtype *bias_data
  CUDA_KERNEL_LOOP(index, count) {
      output_data += index;
      int ow = index % output_width;
      index /= output_width;
      int oh = index % output_height;
      index /= output_height;
      int c = index % channels;
      int n = index / channels;

      int iw = ow * stride_x - pad_x;
      int ih = oh * stride_y - pad_y;

      input_data += ((n * channels + c) * input_height + ih) * input_width + iw;
      weight_data += c * kernel_width * kernel_height;

      Dtype v = 0;
      for (int kh = 0; kh < kernel_height; kh++) {
          if (ih + kh >= 0 && ih + kh < input_height) {
              for (int kw = 0; kw < kernel_width; kw++) {
                  if (iw + kw >= 0 && iw + kw < input_width) {
                      v += input_data[kw] * weight_data[kw];
                  }
              }
          }
          input_data += input_width;
          weight_data += kernel_width;
      }

      // if (bias_data) {
      //     v += bias_data[c];
      // }

      *output_data = v;
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    if (this->group_ == this->channels_){
      // Dtype* bias = NULL;
      // if (this->bias_term_)
      //   bias = this->blobs_[1]->gpu_data();
      int dim = top[i]->count();
      spatial_conv_fwd_kernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
            dim, this->channels_, top[i]->shape(2), top[i]->shape(3), 
            bottom[i]->shape(2), bottom[i]->shape(3),
            this->blobs_[0]->shape(2), this->blobs_[0]->shape(3),
            this->stride_.cpu_data()[0], this->stride_.cpu_data()[1], this->pad_.cpu_data()[0], this->pad_.cpu_data()[1], 
            bottom_data, weight, top_data);
    }
    else{
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
}

template <typename Dtype>
__global__ void spatial_conv_bwd_feature_kernel(const int count, const int channels, const int output_height, const int output_width,
        const int input_height, const int input_width, 
        const int kernel_height, const int kernel_width,
        const int stride_y, const int stride_x, const int pad_y, const int pad_x,
        const Dtype *weight_data, const Dtype *diff,
        Dtype *fea_diff, const Dtype scale_target) {

    CUDA_KERNEL_LOOP(index, count) {
        fea_diff += index;
        const int iw = index % input_width;
        index /= input_width;
        const int ih = index % input_height;
        index /= input_height;
        const int c = index % channels;
        const int n = index / channels;
        
        const int base_ow = min((iw + pad_x) / stride_x, output_width - 1);
        const int base_oh = min((ih + pad_y) / stride_y, output_height - 1);

        diff += (n * channels + c) * output_height * output_width;
        weight_data += c * kernel_height * kernel_width;

        Dtype v = 0;
        for (int oh = base_oh; oh >= 0; oh--) {
            int kh = ih - (oh * stride_y - pad_y);
            if (kh < kernel_height) {
                for (int ow = base_ow; ow >= 0; ow--) {
                    int kw = iw - (ow * stride_x - pad_x);
                    if (kw < kernel_width) {
                        v += weight_data[kh * kernel_width + kw] * diff[oh * output_width + ow];
                    }
                    else {
                        break;
                    }
                }
            }
            else {
                break;
            }
        }
        
        if (scale_target == 0) {
            *fea_diff = v;
        }
        else {
            *fea_diff = (*fea_diff) * scale_target + v;
        }
    }
}


template <typename Dtype>
__global__ void spatial_conv_bwd_weight_kernel(const int count, const int num, const int channels, 
        const int output_height, const int output_width,
        const int input_height, const int input_width, 
        const int kernel_height, const int kernel_width,
        const int stride_y, const int stride_x, const int pad_y, const int pad_x,
        const Dtype *input_data, const Dtype *diff,
        Dtype *weight_diff) {
    CUDA_KERNEL_LOOP(index, count) {
        weight_diff += index;
        const int kw = index % kernel_width;
        index /= kernel_width;
        const int kh = index % kernel_height;
        index /= kernel_height;
        const int c = index % channels;
        index /= channels;
        const int ow = index % output_width;
        const int oh = index / output_width;

        const int iw = ow * stride_x - pad_x + kw;
        const int ih = oh * stride_y - pad_y + kh;

        Dtype v = 0;
        if (iw >= 0 && iw < input_width && ih >= 0 && ih < input_height) {
            input_data += (c * input_height + ih) * input_width + iw;
            diff += (c * output_height + oh) * output_width + ow;
            const int input_stride = channels * input_height * input_width;
            const int output_stride = channels * output_height * output_width;

            for (int n = 0; n < num; n++) {
                v += input_data[input_stride * n] * diff[output_stride * n];
            }
        }
        *weight_diff = v;
    }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
      if (this->group_ == this->channels_) {
        int dim = top[i]->count();
        sum_multiplier_.Reshape(1,1,top[0]->shape(2), top[0]->shape(3));
        caffe_set(top[0]->shape(2) * top[0]->shape(3), Dtype(1), sum_multiplier_.mutable_cpu_data());
        if (this->param_propagate_down_[0]) {
          int count = dim/this->num_ * this->blobs_[0]->shape(2) * this->blobs_[0]->shape(3);
          buffer.Reshape(1,1,1,count);
          spatial_conv_bwd_weight_kernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                count, this->num_, this->channels_, 
                top[i]->shape(2), top[i]->shape(3), 
                bottom[i]->shape(2), bottom[i]->shape(3),
                this->blobs_[0]->shape(2), this->blobs_[0]->shape(3),
                this->stride_.cpu_data()[0], this->stride_.cpu_data()[1], this->pad_.cpu_data()[0], this->pad_.cpu_data()[1], 
                bottom_data, top_diff, buffer.mutable_gpu_data());
          int spatial_size = top[i]->shape(2) * top[i]->shape(3);
          caffe_gpu_gemv<Dtype>(CblasTrans, spatial_size, count / spatial_size, (Dtype)1.,
              buffer.gpu_data(), sum_multiplier_.gpu_data(), (Dtype)0., weight_diff);
        }
        if (propagate_down[i]) {
          spatial_conv_bwd_feature_kernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
                dim, this->channels_, 
                top[i]->shape(2), top[i]->shape(3), 
                bottom[i]->shape(2), bottom[i]->shape(3),
                this->blobs_[0]->shape(2), this->blobs_[0]->shape(3),
                this->stride_.cpu_data()[0], this->stride_.cpu_data()[1], this->pad_.cpu_data()[0], this->pad_.cpu_data()[1], 
                weight, top_diff, bottom_diff, 0);
        }
      } else {
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
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe