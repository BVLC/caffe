// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int bin_num_h, const int bin_num_w,
    const float bin_size_h, const float bin_size_w, Dtype* top_data,
    int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % bin_num_w;
    int ph = (index / bin_num_w) % bin_num_h;
    int c = (index / bin_num_w / bin_num_h) % channels;
    int n = index / bin_num_w / bin_num_h / channels;
    int hstart = max(floor(ph * bin_size_h), 0);
    int wstart = max(floor(pw * bin_size_w), 0);
    int hend = min(ceil((ph + 1) * bin_size_h), height);
    int wend = min(ceil((pw + 1) * bin_size_w), width);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_data[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int bin_num_h, const int bin_num_w,
    const float bin_size_h, const float bin_size_w, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % bin_num_w;
    int ph = (index / bin_num_w) % bin_num_h;
    int c = (index / bin_num_w / bin_num_h) % channels;
    int n = index / bin_num_w / bin_num_h / channels;
    int hstart = max(floor(ph * bin_size_h), 0);
    int wstart = max(floor(pw * bin_size_w), 0);
    int hend = min(ceil((ph + 1) * bin_size_h), height);
    int wend = min(ceil((pw + 1) * bin_size_w), width);
    int pool_size = (hend - hstart) * (wend - wstart);
    Dtype aveval = 0;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_data[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads,
    const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int bin_num_h, const int bin_num_w,a
    const float bin_size_h, const float bin_size_w,
    Dtype* rand_idx, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % bin_num_w;
    int ph = (index / bin_num_w) % bin_num_h;
    int c = (index / bin_num_w / bin_num_h) % channels;
    int n = index / bin_num_w / bin_num_h / channels;
    int hstart = max(floor(ph * bin_size_h), 0);
    int wstart = max(floor(pw * bin_size_w), 0);
    int hend = min(ceil((ph + 1) * bin_size_h), height);
    int wend = min(ceil((pw + 1) * bin_size_w), width);
    Dtype cumsum = 0.;
    bottom_data += (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
      }
    }
    float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_data[h * width + w];
          return;
        }
      }
    }
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads,
    const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int bin_num_h, const int bin_num_w,
    const float bin_size_h, const float bin_size_w, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % bin_num_w;
    int ph = (index / bin_num_w) % bin_num_h;
    int c = (index / bin_num_w / bin_num_h) % channels;
    int n = index / bin_num_w / bin_num_h / channels;
    int hstart = max(floor(ph * bin_size_h), 0);
    int wstart = max(floor(pw * bin_size_w), 0);
    int hend = min(ceil((ph + 1) * bin_size_h), height);
    int wend = min(ceil((pw + 1) * bin_size_w), width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    bottom_data += (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
        cumvalues += bottom_data[h * width + w] * bottom_data[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}


template <typename Dtype>
Dtype PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  int count = (*top)[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top->size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = (*top)[1]->mutable_gpu_data();
    } else {
      mask = max_idx_->mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, bin_num_h_, bin_num_w_, bin_size_h_, bin_size_w_,
        top_data, mask, top_mask);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, bin_num_h_, bin_num_w_, bin_size_h_, bin_size_w_,
        top_data);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (Caffe::phase() == Caffe::TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, bin_num_h_, bin_num_w_, bin_size_h_, bin_size_w_,
          rand_idx_.mutable_gpu_data(), top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, bin_num_h_, bin_num_w_, bin_size_h_, bin_size_w_,
          top_data);
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
  return Dtype(0.);
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const Dtype* top_mask, const int num, const int channels,
    const int height, const int width, const int bin_num_h,
    const int bin_num_w, const float bin_size_h, const float bin_size_w,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = max(floor(h / bin_size_h - 1), 0);
    int phend = min(ceil((h + 1) / bin_size_h), bin_num_h);
    int pwstart = max(floor(w / bin_size_w - 1), 0);
    int pwend = min(ceil((w + 1) / bin_size_w), bin_num_w);
    Dtype gradient = 0;
    int offset = (n * channels + c) * bin_num_h * bin_num_w;
    top_diff += offset;
    if (mask) {
      mask += offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask[ph * bin_num_w + pw] == h * width + w) {
            gradient += top_diff[ph * bin_num_w + pw];
          }
        }
      }
    } else {
      top_mask += offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask[ph * bin_num_w + pw] == h * width + w) {
            gradient += top_diff[ph * bin_num_w + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int bin_num_h, const int bin_num_w,
    const float bin_size_h, const float bin_size_w, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = max(floor(h / bin_size_h - 1), 0);
    int phend = min(ceil((h + 1) / bin_size_h), bin_num_h);
    int pwstart = max(floor(w / bin_size_w - 1), 0);
    int pwend = min(ceil((w + 1) / bin_size_w), bin_num_w);
    Dtype gradient = 0;
    top_diff += (n * channels + c) * bin_num_h * bin_num_w;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = max(floor(ph * bin_size_h), 0);
        int wstart = max(floor(pw * bin_size_w), 0);
        int hend = min(ceil((ph + 1) * bin_size_h), height);
        int wend = min(ceil((pw + 1) * bin_size_w), width);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff[ph * bin_num_w + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads,
    const Dtype* rand_idx, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int bin_num_h, const int bin_num_w,
    const float bin_size_h, const float bin_size_w, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = max(floor(h / bin_size_h - 1), 0);
    int phend = min(ceil((h + 1) / bin_size_h), bin_num_h);
    int pwstart = max(floor(w / bin_size_w - 1), 0);
    int pwend = min(ceil((w + 1) / bin_size_w), bin_num_w);
    Dtype gradient = 0;
    rand_idx += (n * channels + c) * bin_num_h * bin_num_w;
    top_diff += (n * channels + c) * bin_num_h * bin_num_w;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff[ph * bin_num_w + pw] *
            (index == static_cast<int>(rand_idx[ph * bin_num_w + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  const int count = (*bottom)[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_->gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, bin_num_h_, bin_num_w_,
        bin_size_h_, bin_size_w_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, bin_num_h_, bin_num_w_,
        bin_size_h_, bin_size_w_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff,
        top[0]->num(), channels_, height_, width_, bin_num_h_,
        bin_num_w_, bin_size_h_, bin_size_w_, bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_CLASS(PoolingLayer);


}  // namespace caffe
