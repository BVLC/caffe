#include <vector>
#include "caffe/layers/conv_dw_layer.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void ConvolutionDepthwiseWeightForward(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const weight_data, const int num, const int channels,
    const int top_height, const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / channels / top_height / top_width;
    const int c = (index / top_height / top_width) % channels;
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const Dtype* weight = weight_data + c * kernel_h * kernel_w;
    Dtype value = 0;
    for (int kh = 0; kh < kernel_h; ++kh)
    {
      for (int kw = 0; kw < kernel_w; ++kw)
      {
        const int h_in = -pad_h + h * stride_h + kh * dilation_h;
        const int w_in = -pad_w + w * stride_w + kw * dilation_w;
        if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width))
        {
          const int offset = ((n * channels + c) * bottom_height + h_in) * bottom_width + w_in;
          value += (*weight) * bottom_data[offset];
        }
        ++weight;
      }
    }
    top_data[index] = value;
  }
}

template <typename Dtype>
__global__ void ConvolutionDepthwiseBiasForward(const int nthreads,
    const Dtype* const bias_data, const int num, const int channels,
    const int top_height, const int top_width, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = (index / top_height / top_width) % channels;
    top_data[index] += bias_data[c];
  }
}

template <typename Dtype>
void ConvolutionDepthwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight_data = this->blobs_[0]->gpu_data();
  const int count = top[0]->count();
  const int num = top[0]->num();
  const int channels = top[0]->channels();
  const int top_height = top[0]->height();
  const int top_width = top[0]->width();
  const int bottom_height = bottom[0]->height();
  const int bottom_width = bottom[0]->width();
  ConvolutionDepthwiseWeightForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, weight_data, num, channels,
      top_height, top_width, bottom_height, bottom_width,
      kernel_h_, kernel_w_, stride_h_, stride_w_,
      pad_h_, pad_w_, dilation_h_, dilation_w_, top_data);
  if (this->layer_param_.convolution_param().bias_term())
  {
    const Dtype* bias_data = this->blobs_[1]->gpu_data();
    ConvolutionDepthwiseBiasForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bias_data, num, channels,
        top_height, top_width, top_data);
  }
}

template <typename Dtype>
__global__ void ConvolutionDepthwiseWeightBackward(const int nthreads,
    const Dtype* const top_diff, const Dtype* const bottom_data, const int num, const int channels,
    const int top_height, const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
    Dtype* const buffer_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const int kh = (index / kernel_w / num / top_height / top_width) % kernel_h;
    const int kw = (index / num / top_height / top_width) % kernel_w;
    const int h_in = -pad_h + h * stride_h + kh * dilation_h;
    const int w_in = -pad_w + w * stride_w + kw * dilation_w;
    if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width))
    {
      const int c = index / kernel_h / kernel_w / num / top_height / top_width;
      const int n = (index / top_height / top_width) % num;
      const int top_offset = ((n * channels + c) * top_height + h) * top_width + w;
      const int bottom_offset = ((n * channels + c) * bottom_height + h_in) * bottom_width + w_in;
      buffer_data[index] = top_diff[top_offset] * bottom_data[bottom_offset];
    }
    else
    {
      buffer_data[index] = 0;
    }
  }
}

template <typename Dtype>
__global__ void ConvolutionDepthwiseBottomBackward(const int nthreads,
    const Dtype* const top_diff, const Dtype* const weight_data, const int num, const int channels,
    const int top_height, const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / channels / bottom_height / bottom_width;
    const int c = (index / bottom_height / bottom_width) % channels;
    const int h = (index / bottom_width) % bottom_height;
    const int w = index % bottom_width;
    const Dtype* weight = weight_data + c * kernel_h * kernel_w;
    Dtype value = 0;
    for (int kh = 0; kh < kernel_h; ++kh)
    {
      for (int kw = 0; kw < kernel_w; ++kw)
      {
        const int h_out_s = h + pad_h - kh * dilation_h;
        const int w_out_s = w + pad_w - kw * dilation_w;
        if (((h_out_s % stride_h) == 0) && ((w_out_s % stride_w) == 0))
        {
          const int h_out = h_out_s / stride_h;
          const int w_out = w_out_s / stride_w;
          if ((h_out >= 0) && (h_out < top_height) && (w_out >= 0) && (w_out < top_width))
          {
            const int offset = ((n * channels + c) * top_height + h_out) * top_width + w_out;
            value += (*weight) * top_diff[offset];
          }
        }
        ++weight;
      }
    }
    bottom_diff[index] += value;
  }
}

template <typename Dtype>
__global__ void ConvolutionDepthwiseBiasBackward(const int nthreads,
    const Dtype* const top_diff, const int num, const int channels,
    const int top_height, const int top_width, Dtype* const buffer_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = index / num / top_height / top_width;
    const int n = (index / top_height / top_width) % num;
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const int offset = ((n * channels + c) * top_height + h) * top_width + w;
    buffer_data[index] = top_diff[offset];
  }
}

template <typename Dtype>
void ConvolutionDepthwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const int bottom_count = bottom[0]->count();
  const int num = top[0]->num();
  const int channels = top[0]->channels();
  const int top_height = top[0]->height();
  const int top_width = top[0]->width();
  const int bottom_height = bottom[0]->height();
  const int bottom_width = bottom[0]->width();
  const int length = num * top_height * top_width;
  caffe_gpu_set(bottom_count, Dtype(0), bottom[0]->mutable_gpu_diff());
  if (this->layer_param_.convolution_param().bias_term() && this->param_propagate_down_[1])
  {
    const int bias_buffer_count = bias_buffer_.count();
    Dtype* bias_buffer_mutable_data = bias_buffer_.mutable_gpu_data();
    ConvolutionDepthwiseBiasBackward<Dtype><<<CAFFE_GET_BLOCKS(bias_buffer_count), CAFFE_CUDA_NUM_THREADS>>>(
        bias_buffer_count, top_diff, num, channels,
        top_height, top_width, bias_buffer_mutable_data);
    const int bias_count = this->blobs_[1]->count();
    const Dtype* bias_buffer_data = bias_buffer_.gpu_data();
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    const Dtype* bias_multiplier_data = bias_multiplier_.gpu_data();
    caffe_gpu_gemv(CblasNoTrans, bias_count, length, Dtype(1), bias_buffer_data, bias_multiplier_data, Dtype(1), bias_diff);
  }
  if (this->param_propagate_down_[0])
  {
    const int weight_buffer_count = weight_buffer_.count();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* weight_buffer_mutable_data = weight_buffer_.mutable_gpu_data();
    ConvolutionDepthwiseWeightBackward<Dtype><<<CAFFE_GET_BLOCKS(weight_buffer_count), CAFFE_CUDA_NUM_THREADS>>>(
        weight_buffer_count, top_diff, bottom_data, num, channels,
        top_height, top_width, bottom_height, bottom_width,
        kernel_h_, kernel_w_, stride_h_, stride_w_,
        pad_h_, pad_w_, dilation_h_, dilation_w_, weight_buffer_mutable_data);
    const int weight_count = this->blobs_[0]->count();
    const Dtype* weight_buffer_data = weight_buffer_.gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    const Dtype* weight_multiplier_data = weight_multiplier_.gpu_data();
    caffe_gpu_gemv(CblasNoTrans, weight_count, length, Dtype(1), weight_buffer_data, weight_multiplier_data, Dtype(1), weight_diff);
  }
  if (propagate_down[0])
  {
    const Dtype* weight_data = this->blobs_[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    ConvolutionDepthwiseBottomBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
        bottom_count, top_diff, weight_data, num, channels,
        top_height, top_width, bottom_height, bottom_width,
        kernel_h_, kernel_w_, stride_h_, stride_w_,
        pad_h_, pad_w_, dilation_h_, dilation_w_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionDepthwiseLayer);

}  // namespace caffe
