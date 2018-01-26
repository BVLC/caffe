#include <algorithm>
#include <cfloat>
#include <vector>
#include <string>

#include <unistd.h>
#include <stdio.h>
#include "caffe/blob.hpp"

#include "caffe/layers/saliency_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SalPoolForward_SaliencyWeighting(const int nthreads,
    const Dtype* const image_data, const Dtype* const saliency_data,
    const int num, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    //const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    float aveval = 0;
    float salval = 0;
    const Dtype* const bottom_slice = image_data + (n * channels + c) * height * width;
    const Dtype* const saliency_bottom_slice = saliency_data + (n * channels + c) * height * width;

    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w] * saliency_bottom_slice[h * width + w];
        //aveval += bottom_slice[h * width + w];
        salval += saliency_bottom_slice[h * width + w];
      }
    }
    //printf("Index:%d \t Eval:%f \t Salval:%f\n", index, aveval, salval);
    if (salval == 0) {
      top_data[index] = 0;
    } else {
      top_data[index] = aveval / salval;
    }
  }
}

template <typename Dtype>
__global__ void SalPoolForward_Random_Sampling(const int nthreads,
  const Dtype* const image_data, const Dtype* const saliency_data,
  const int num, float* numbers, const int channels, const int height, const int width,
  const int pooled_height, const int pooled_width, const int kernel_h,
  const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
  const int pad_w, Dtype* const top_data) {
CUDA_KERNEL_LOOP(index, nthreads) {
  const int pw = index % pooled_width;
  const int ph = (index / pooled_width) % pooled_height;
  const int c = (index / pooled_width / pooled_height) % channels;
  const int n = index / pooled_width / pooled_height / channels;
  int hstart = ph * stride_h - pad_h;
  int wstart = pw * stride_w - pad_w;
  int hend = min(hstart + kernel_h, height + pad_h);
  int wend = min(wstart + kernel_w, width + pad_w);
  //const int pool_size = (hend - hstart) * (wend - wstart);
  hstart = max(hstart, 0);
  wstart = max(wstart, 0);
  hend = min(hend, height);
  wend = min(wend, width);

  const Dtype* const bottom_slice = image_data + (n * channels + c) * height * width;
  const Dtype* const saliency_bottom_slice = saliency_data + (n * channels + c) * height * width;

  // Weibull distribution
  float lambda = 0.5;
  float k = 4.0;

  float Ps = lambda * pow(-log(1-numbers[index]), (1/k));

  // Saliency value at index position
  float salval = saliency_bottom_slice[((hend-hstart)/2)+hstart * width + ((wend-wstart)/2)+wstart];

  if (Ps < salval){
    // Compute MaxPooling
    float maxval = -FLT_MAX;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval){
          maxval = bottom_slice[h * width + w];
        }
      }
    }
    top_data[index] = maxval;
    //top_data[index] = 1.0;
    //printf("Index:%d \t MaxVal:%f \t Salval:%d, Rand:%f\t (Max)\n", index, maxval, salval, Ps);
  }
  else{
    // Compute min val
    float minval = FLT_MAX;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] < minval){
          minval = bottom_slice[h * width + w];
        }
      }
    }
    top_data[index] = minval;
    //top_data[index] = 0.0;
    //printf("Index:%d \t AveVal:%f \t Salval:%f, Rand:%f\t (Average)\n", index, minval, salval, Ps);
    }
  }
}

template <typename Dtype>
__global__ void SalPoolForward_Random_Sampling_Weighting(const int nthreads,
  const Dtype* const image_data, const Dtype* const saliency_data,
  const int num, float* numbers, const int channels, const int height, const int width,
  const int pooled_height, const int pooled_width, const int kernel_h,
  const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
  const int pad_w, Dtype* const top_data) {
CUDA_KERNEL_LOOP(index, nthreads) {
  const int pw = index % pooled_width;
  const int ph = (index / pooled_width) % pooled_height;
  const int c = (index / pooled_width / pooled_height) % channels;
  const int n = index / pooled_width / pooled_height / channels;
  int hstart = ph * stride_h - pad_h;
  int wstart = pw * stride_w - pad_w;
  int hend = min(hstart + kernel_h, height + pad_h);
  int wend = min(wstart + kernel_w, width + pad_w);
  //const int pool_size = (hend - hstart) * (wend - wstart);
  hstart = max(hstart, 0);
  wstart = max(wstart, 0);
  hend = min(hend, height);
  wend = min(wend, width);

  const Dtype* const bottom_slice = image_data + (n * channels + c) * height * width;
  const Dtype* const saliency_bottom_slice = saliency_data + (n * channels + c) * height * width;

  // Weibull distribution
  float lambda = 0.5;
  float k = 4.0;

  float Ps = lambda * pow(-log(1-numbers[index]), (1/k));

  // Saliency value at index position
  float salval = saliency_bottom_slice[((hend-hstart)/2)+hstart * width + ((wend-wstart)/2)+wstart];

  if (Ps <= salval){
    // Compute MaxPooling
    float maxval = -FLT_MAX;
    int maxvalidx = -1;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval){
          maxval = bottom_slice[h * width + w];
          maxvalidx = h * width + w;
        }
      }
    }
    top_data[index] = maxval * saliency_bottom_slice[maxvalidx];
    //top_data[index] = 1.0;
    //printf("Index:%d \t MaxVal:%f \t Salval:%d, Rand:%f\t (Max)\n", index, maxval, salval, Ps);
  }
  else{
    // Compute min val
    float minval = FLT_MAX;
    int minvalidx = -1;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] < minval){
          minval = bottom_slice[h * width + w];
          minvalidx = h * width + w;
        }
      }
    }
    top_data[index] = minval * saliency_bottom_slice[minvalidx];
    //top_data[index] = 0.0;
    //printf("Index:%d \t AveVal:%f \t Salval:%f, Rand:%f\t (Average)\n", index, minval, salval, Ps);
    }
  }
}

template <typename Dtype>
__global__ void AveragePoolForward(const int nthreads,
    const Dtype* const image_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        image_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

template <typename Dtype>
void SaliencyPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top){
    const Dtype* image_data = bottom[0]->gpu_data();
    const Dtype* saliency_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int count = top[0]->count();

    // Generate random numbers
    const float lower = 0.0;
    const float upper = 1.0;
    Blob<float>* rands = new Blob<float>(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
    caffe_gpu_rng_uniform(bottom[0]->count(), lower, upper, rands->mutable_gpu_data());

    int PoolingMethod = 1;
    // 0 = Saliency Weighting     (SAL: SalWeighting,                         NON-SAL: Zero)
    // 1 = RandomSampling         (SAL: MaxPooling,                           NON-SAL: Mean value) - Using Ps (Weibull distribution)
    // 2 = MaxPooling*Weighting   (SAL: MaxValue*SalientValue                 NON-SAL: Mean value)
    // 3 = Average Pooling        (Average Pooling for each value)

    switch (PoolingMethod) {
      case 0:
        // CUDA Routine for SalPoolForward_SaliencyWeighting
        SalPoolForward_SaliencyWeighting<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, image_data, saliency_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
      break;
      case 1:
        // CUDA Routine for SalPoolForward_Random_Sampling
        SalPoolForward_Random_Sampling<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, image_data, saliency_data, bottom[0]->num(), rands->mutable_gpu_data(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
      break;
      case 2:
        // CUDA Routine for SalPoolForward_Random_Sampling_Weighting
        SalPoolForward_Random_Sampling_Weighting<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, image_data, saliency_data, bottom[0]->num(), rands->mutable_gpu_data(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
      break;
      case 3:
      // CUDA Routine for Average Pooling
        AveragePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, image_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
      break;
      default:
        LOG(FATAL) << "Unknown pooling method.";
    }
  CUDA_POST_KERNEL_CHECK;
  delete rands;
}

template <typename Dtype>
__global__ void SalPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void SaliencyPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
  /*
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);

  // CUDA Routine
  SalPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, top_diff, top[0]->num(), channels_,
    height_, width_, pooled_height_, pooled_width_, kernel_h_,
    kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  */
}

INSTANTIATE_LAYER_GPU_FUNCS(SaliencyPoolingLayer);

}  // namespace caffe
