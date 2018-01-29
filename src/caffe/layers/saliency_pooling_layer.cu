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
      const int kernel_w, const int stride_h, const int stride_w, Dtype* const top_data,
      int* mask, Dtype* top_mask) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int pw = index % pooled_width;
      const int ph = (index / pooled_width) % pooled_height;
      const int c = (index / pooled_width / pooled_height) % channels;
      const int n = index / pooled_width / pooled_height / channels;
      int hstart = ph * stride_h;
      int wstart = pw * stride_w;
      int hend = min(hstart + kernel_h, height);
      int wend = min(wstart + kernel_w, width);
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
      if (mask) {
        mask[index] = ((hend-hstart)/2)+hstart * width + ((wend-wstart)/2)+wstart;
      } else {
        top_mask[index] = ((hend-hstart)/2)+hstart * width + ((wend-wstart)/2)+wstart;
      }
    }
  }

template <typename Dtype>
__global__ void SalPoolForward_Random_Sampling(const int nthreads,
  const Dtype* const image_data, const Dtype* const saliency_data,
  const int num, float* numbers, const int channels, const int height, const int width,
  const int pooled_height, const int pooled_width, const int kernel_h,
  const int kernel_w, const int stride_h, const int stride_w, Dtype* const top_data,
  int* mask, Dtype* top_mask) {
CUDA_KERNEL_LOOP(index, nthreads) {
  const int pw = index % pooled_width;
  const int ph = (index / pooled_width) % pooled_height;
  const int c = (index / pooled_width / pooled_height) % channels;
  const int n = index / pooled_width / pooled_height / channels;
  int hstart = ph * stride_h;
  int wstart = pw * stride_w;
  int hend = min(hstart + kernel_h, height);
  int wend = min(wstart + kernel_w, width);
  //const int pool_size = (hend - hstart) * (wend - wstart);
  hstart = max(hstart, 0);
  wstart = max(wstart, 0);
  hend = min(hend, height);
  wend = min(wend, width);

  const Dtype* const image_bottom_slice = image_data + (n * channels + c) * height * width;
  const Dtype* const saliency_bottom_slice = saliency_data + (n * channels + c) * height * width;

  // Weibull distribution
  float lambda = 0.5;
  float k = 4.0;

  Dtype Ps = lambda * pow(-log(1-numbers[index]), (1/k));

  // Saliency value at index position
  Dtype salval = saliency_bottom_slice[((hend-hstart)/2)+hstart * width + ((wend-wstart)/2)+wstart];
  int validx = -1;
  if (Ps < salval){
    // Compute MaxPooling
    Dtype maxval = -FLT_MAX;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (image_bottom_slice[h * width + w] > maxval){
          maxval = image_bottom_slice[h * width + w];
          validx = h * width + w;
        }
      }
    }
    top_data[index] = maxval; // * saliency_bottom_slice[validx];
    //printf("Index:%d \t MaxVal:%f \t Salval:%d, Rand:%f\t (Max)\n", index, maxval, salval, Ps);
  }
  else{
    // Compute min val
    float minval = FLT_MAX;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (image_bottom_slice[h * width + w] < minval){
          minval = image_bottom_slice[h * width + w];
          validx =  h * width + w;
        }
      }
    }
    top_data[index] = minval;
    //printf("Index:%d \t AveVal:%f \t Salval:%f, Rand:%f\t (Average)\n", index, minval, salval, Ps);
  }
  if (mask) {
    mask[index] = validx;
  } else {
    top_mask[index] = validx;
  }
  }
}

template <typename Dtype>
__global__ void SalPoolForward_Random_Sampling_Weighting(const int nthreads,
  const Dtype* const image_data, const Dtype* const saliency_data,
  const int num, float* numbers, const int channels, const int height, const int width,
  const int pooled_height, const int pooled_width, const int kernel_h,
  const int kernel_w, const int stride_h, const int stride_w, Dtype* const top_data,
  int* mask, Dtype* top_mask) {
CUDA_KERNEL_LOOP(index, nthreads) {
  const int pw = index % pooled_width;
  const int ph = (index / pooled_width) % pooled_height;
  const int c = (index / pooled_width / pooled_height) % channels;
  const int n = index / pooled_width / pooled_height / channels;
  int hstart = ph * stride_h;
  int wstart = pw * stride_w;
  int hend = min(hstart + kernel_h, height);
  int wend = min(wstart + kernel_w, width);
  //const int pool_size = (hend - hstart) * (wend - wstart);
  hstart = max(hstart, 0);
  wstart = max(wstart, 0);
  hend = min(hend, height);
  wend = min(wend, width);

  const Dtype* const image_bottom_slice = image_data + (n * channels + c) * height * width;
  const Dtype* const saliency_bottom_slice = saliency_data + (n * channels + c) * height * width;

  // Weibull distribution
  float lambda = 0.5;
  float k = 4.0;

  Dtype Ps = lambda * pow(-log(1-numbers[index]), (1/k));

  // Saliency value at index position
  Dtype salval = saliency_bottom_slice[((hend-hstart)/2)+hstart * width + ((wend-wstart)/2)+wstart];
  int validx = -1;
  if (Ps < salval){
    // Compute MaxPooling
    Dtype maxval = -FLT_MAX;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (image_bottom_slice[h * width + w] > maxval){
          maxval = image_bottom_slice[h * width + w];
          validx = h * width + w;
        }
      }
    }
    top_data[index] = maxval * saliency_bottom_slice[validx];
    //printf("Index:%d \t MaxVal:%f \t Salval:%d, Rand:%f\t (Max)\n", index, maxval, salval, Ps);
  }
  else{
    // Compute min val
    float minval = FLT_MAX;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (image_bottom_slice[h * width + w] < minval){
          minval = image_bottom_slice[h * width + w];
          validx =  h * width + w;
        }
      }
    }
    top_data[index] = minval * saliency_bottom_slice[validx];;
    //printf("Index:%d \t AveVal:%f \t Salval:%f, Rand:%f\t (Average)\n", index, minval, salval, Ps);
  }
  if (mask) {
    mask[index] = validx;
  } else {
    top_mask[index] = validx;
  }
  }
}

template <typename Dtype>
void SaliencyPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top){
    const Dtype* image_data = bottom[0]->gpu_data();
    const Dtype* saliency_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int count = top[0]->count();

    // We'll output the mask to top[1] if it's of size >1.
    const bool use_top_mask = top.size() > 1;
    int* mask = NULL;
    Dtype* top_mask = NULL;

    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }

    // Generate random numbers
    const float lower = 0.0;
    const float upper = 1.0;
    //Blob<float>* rands = new Blob<float>(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
    caffe_gpu_rng_uniform(bottom[0]->count(), lower, upper, randoms_.mutable_gpu_data());

    // 0 = Saliency Weighting           (SAL: Feat*Salvals,             NON-SAL: Zero)
    // 1 = RandomSampling               (SAL: MaxPooling,               NON-SAL: Min value) - Using Ps (Weibull distribution)
    // 2 = RandomSampling + Weighting   (SAL: MaxPooling*SalientValue   NON-SAL: Min value) - Using Ps (Weibull distribution)
    switch (PoolMethod) {
      case 0:
      // CUDA Routine for SalPoolForward_Random_Sampling
      SalPoolForward_SaliencyWeighting<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, image_data, saliency_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, top_data, mask, top_mask);
      break;
      case 1:
      // CUDA Routine for SalPoolForward_Random_Sampling
      SalPoolForward_Random_Sampling<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, image_data, saliency_data, bottom[0]->num(), randoms_.mutable_gpu_data(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, top_data, mask, top_mask);
      break;
      case 2:
      // CUDA Routine for SalPoolForward_Random_Sampling
      SalPoolForward_Random_Sampling_Weighting<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, image_data, saliency_data, bottom[0]->num(), randoms_.mutable_gpu_data(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, top_data, mask, top_mask);
      break;
      default:
        LOG(FATAL) << "Unknown pooling method.";
    }
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void SalPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h  < kernel_h) ? 0 : (h  - kernel_h) / stride_h + 1;
    const int phend = min((h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w  < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min((w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int* const mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    } else {
      const Dtype* const top_mask_slice = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void SaliencyPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (!propagate_down[0]) {
    return;
  }

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);

  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;

  if (use_top_mask) {
    top_mask = top[1]->gpu_data();
  } else {
    mask = max_idx_.gpu_data();
  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  SalPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, mask, top_mask, top[0]->num(), channels_,
      height_, width_, pooled_height_, pooled_width_,
      kernel_h_, kernel_w_, stride_h_, stride_w_,
      bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SaliencyPoolingLayer);

}  // namespace caffe
