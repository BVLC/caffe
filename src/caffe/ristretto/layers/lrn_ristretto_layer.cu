#include <vector>

#include "caffe/layers/lrn_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "ristretto/base_ristretto_layer.hpp"
#include "ristretto/base_ristretto_layer.cuh"

namespace caffe {

// Device function wrapper for quantization to minifloat numbers.
template <typename Dtype>
__device__ Dtype
toFP(Dtype data, const int mant, const int exp, const int index){
  Trim2MiniFloat_device(&data, mant, exp,
      QuantizationParameter_Rounding_NEAREST, index);
  return data;
}

// Same as LRNFillScale, but all intermediate results are quantized to
// minifloat.
template <typename Dtype>
__global__ void LRNFillScaleQ(const int nthreads, const Dtype* const in,
      const int num, const int channels, const int height, const int width,
      const int size, const Dtype alpha_over_size, const Dtype k,
      Dtype* const scale, const int mant, const int exp) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const in_off = in + offset;
    Dtype* const scale_off = scale + offset;
    int head = 0;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      Dtype in_off_q = toFP(in_off[head * step], mant, exp, index);
      accum_scale += toFP(in_off_q * in_off_q, mant, exp, index);
      accum_scale = toFP(accum_scale, mant, exp, index);
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      Dtype in_off_q = toFP(in_off[head * step], mant, exp, index);
      accum_scale += toFP(in_off_q * in_off_q, mant, exp, index);
      accum_scale = toFP(accum_scale, mant, exp, index);
      if (head - size >= 0) {
        Dtype in_off_q = toFP(in_off[(head - size) * step], mant, exp, index);
        accum_scale -= toFP(in_off_q * in_off_q, mant, exp, index);
        accum_scale = toFP(accum_scale, mant, exp, index);
      }
      Dtype tmp = toFP(accum_scale * alpha_over_size, mant, exp, index);
      tmp = toFP(k + tmp, mant, exp, index);
      scale_off[(head - post_pad) * step] = tmp;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        Dtype in_off_q = toFP(in_off[(head - size) * step], mant, exp, index);
        accum_scale -= toFP(in_off_q * in_off_q, mant, exp, index);
        accum_scale = toFP(accum_scale, mant, exp, index);
      }
      Dtype tmp = toFP(accum_scale * alpha_over_size, mant, exp, index);
      tmp = toFP(k + tmp, mant, exp, index);
      scale_off[(head - post_pad) * step] = tmp;
      ++head;
    }
  }
}

template <typename Dtype>
void LRNRistrettoLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelForward_gpu(bottom, top);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    //TODO implement
    LOG(FATAL) << "Unsupported normalization region: " <<
        LRNParameter_NormRegion_WITHIN_CHANNEL;
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

// Same as LRNComputeOutput, but all intermediate results are quantized to
// minifloat
// TODO: check if it would be faster to just put it into the previous kernel.
template <typename Dtype>
__global__ void LRNComputeOutputQ(const int nthreads, const Dtype* const in,
      const Dtype* const scale, const Dtype negative_beta, Dtype* const out,
      const int mant, const int exp) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype in_q = toFP(in[index], mant, exp, index);
    Dtype pow_q = toFP(pow(scale[index], negative_beta), mant, exp, index);
    out[index] = toFP(in_q * pow_q, mant, exp, index);
  }
}

template <typename Dtype>
void LRNRistrettoLayer<Dtype>::CrossChannelForward_gpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, compute scale
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = this->scale_.mutable_gpu_data();
  // We will launch one kernel for each pixel location, and have the kernel
  // go through all the channels.
  int n_threads = this->num_ * this->height_ * this->width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNFillScaleQ<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, this->num_, this->channels_, this->height_,
      this->width_, this->size_, this->alpha_ / this->size_, this->k_,
      scale_data, this->fp_mant_, this->fp_exp_);
  CUDA_POST_KERNEL_CHECK;
  n_threads = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeOutputQ<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, scale_data, -this->beta_, top_data,
      this->fp_mant_, this->fp_exp_);
  CUDA_POST_KERNEL_CHECK;
}
template void LRNRistrettoLayer<float>::CrossChannelForward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void LRNRistrettoLayer<double>::CrossChannelForward_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);

template <typename Dtype>
void LRNRistrettoLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    this->CrossChannelBackward_gpu(top, propagate_down, bottom);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    this->WithinChannelBackward(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LRNRistrettoLayer);

}  // namespace caffe
