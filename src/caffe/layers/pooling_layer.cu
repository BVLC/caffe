#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads,
    const Dtype* bottom_data, const Dtype* bottom_size, const int shrink_factor,
    const int num, const int channels, const int height,
    const int width, const int pool_h, const int pool_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    const int pad_h, const int pad_w, Dtype* top_data,
    int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pool_w;
    int ph = (index / pool_w) % pool_h;
    int c = (index / pool_w / pool_h) % channels;
    int n = index / pool_w / pool_h / channels;
    //
    int img_height = (bottom_size) ? (bottom_size[2 * n] - 1) / shrink_factor + 1 : height;
    int img_width  = (bottom_size) ? (bottom_size[2 * n + 1] - 1) / shrink_factor + 1: width;
    if (bottom_size) {
      kernel_h = img_height / pool_h;
      kernel_w = img_width / pool_w;
      stride_h = kernel_h;
      stride_w = kernel_w;
    }
    //
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, img_height);
    int wend = min(wstart + kernel_w, img_width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
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
__global__ void AvePoolForward(const int nthreads,
    const Dtype* bottom_data, const Dtype* bottom_size, const int shrink_factor,
    const int num, const int channels, const int height,
    const int width, const int pool_h, const int pool_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    const int pad_h, const int pad_w, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pool_w;
    int ph = (index / pool_w) % pool_h;
    int c = (index / pool_w / pool_h) % channels;
    int n = index / pool_w / pool_h / channels;
    //
    int img_height = (bottom_size) ? (bottom_size[2 * n] - 1) / shrink_factor + 1 : height;
    int img_width  = (bottom_size) ? (bottom_size[2 * n + 1] - 1) / shrink_factor + 1: width;
    if (bottom_size) {
      kernel_h = img_height / pool_h;
      kernel_w = img_width / pool_w;
      stride_h = kernel_h;
      stride_w = kernel_w;
    }
    //
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, img_height + pad_h);
    int wend = min(wstart + kernel_w, img_width + pad_w);
    int pool_area = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, img_height);
    wend = min(wend, img_width);
    Dtype aveval = 0;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_data[h * width + w];
      }
    }
    top_data[index] = aveval / pool_area;
  }
}

template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads,
    const Dtype* bottom_data, const Dtype* bottom_size, const int shrink_factor,
    const int num, const int channels, const int height,
    const int width, const int pool_h, const int pool_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    Dtype* rand_idx, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pool_w;
    int ph = (index / pool_w) % pool_h;
    int c = (index / pool_w / pool_h) % channels;
    int n = index / pool_w / pool_h / channels;
    //
    int img_height = (bottom_size) ? (bottom_size[2 * n] - 1) / shrink_factor + 1 : height;
    int img_width  = (bottom_size) ? (bottom_size[2 * n + 1] - 1) / shrink_factor + 1: width;
    if (bottom_size) {
      kernel_h = img_height / pool_h;
      kernel_w = img_width / pool_w;
      stride_h = kernel_h;
      stride_w = kernel_w;
    }
    //
    int hstart = ph * stride_h;
    int wstart = pw * stride_w;
    int hend = min(hstart + kernel_h, img_height);
    int wend = min(wstart + kernel_w, img_width);
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
    const Dtype* bottom_data, const Dtype* bottom_size, const int shrink_factor,
    const int num, const int channels, const int height,
    const int width, const int pool_h, const int pool_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pool_w;
    int ph = (index / pool_w) % pool_h;
    int c = (index / pool_w / pool_h) % channels;
    int n = index / pool_w / pool_h / channels;
    //
    int img_height = (bottom_size) ? (bottom_size[2 * n] - 1) / shrink_factor + 1 : height;
    int img_width  = (bottom_size) ? (bottom_size[2 * n + 1] - 1) / shrink_factor + 1: width;
    if (bottom_size) {
      kernel_h = img_height / pool_h;
      kernel_w = img_width / pool_w;
      stride_h = kernel_h;
      stride_w = kernel_w;
    }
    //
    int hstart = ph * stride_h;
    int wstart = pw * stride_w;
    int hend = min(hstart + kernel_h, img_height);
    int wend = min(wstart + kernel_w, img_width);
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
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_size = (this->layer_param_.pooling_param().do_spm()) ?
    bottom[1]->gpu_data() : 0;
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	count, bottom_data, bottom_size, shrink_factor_, bottom[0]->num(), channels_,
        height_, width_, pool_h_, pool_w_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
        mask, top_mask);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom_size, shrink_factor_, bottom[0]->num(), channels_,
        height_, width_, pool_h_, pool_w_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom_size, shrink_factor_, bottom[0]->num(), channels_,
          height_, width_, pool_h_, pool_w_, kernel_h_,
          kernel_w_, stride_h_, stride_w_,
          rand_idx_.mutable_gpu_data(), top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom_size, shrink_factor_, bottom[0]->num(), channels_,
          height_, width_, pool_h_, pool_w_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, top_data);
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_size, const int shrink_factor,
    const int* mask, const Dtype* top_mask, const int num, const int channels,
    const int height, const int width, const int pool_h, const int pool_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    const int pad_h, const int pad_w, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    //
    int img_height = (bottom_size) ? (bottom_size[2 * n] - 1) / shrink_factor + 1 : height;
    int img_width  = (bottom_size) ? (bottom_size[2 * n + 1] - 1) / shrink_factor + 1: width;
    if (bottom_size) {
      kernel_h = img_height / pool_h;
      kernel_w = img_width / pool_w;
      stride_h = kernel_h;
      stride_w = kernel_w;
    }
    int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    int phend = min((h + pad_h) / stride_h + 1, pool_h);
    int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    int pwend = min((w + pad_w) / stride_w + 1, pool_w);
    Dtype gradient = 0;
    int offset = (n * channels + c) * pool_h * pool_w;
    top_diff += offset;
    if (mask) {
      mask += offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask[ph * pool_w + pw] == h * width + w) {
            gradient += top_diff[ph * pool_w + pw];
          }
        }
      }
    } else {
      top_mask += offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask[ph * pool_w + pw] == h * width + w) {
            gradient += top_diff[ph * pool_w + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_size, const int shrink_factor,
    const int num, const int channels, const int height,
    const int width, const int pool_h, const int pool_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    const int pad_h, const int pad_w,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    //
    int img_height = (bottom_size) ? (bottom_size[2 * n] - 1) / shrink_factor + 1 : height;
    int img_width  = (bottom_size) ? (bottom_size[2 * n + 1] - 1) / shrink_factor + 1: width;
    if (bottom_size) {
      kernel_h = img_height / pool_h;
      kernel_w = img_width / pool_w;
      stride_h = kernel_h;
      stride_w = kernel_w;
    }
    int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int phend = min(h / stride_h + 1, pool_h);
    int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    int pwend = min(w / stride_w + 1, pool_w);
    Dtype gradient = 0;
    top_diff += (n * channels + c) * pool_h * pool_w;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_area = (hend - hstart) * (wend - wstart);
        gradient += top_diff[ph * pool_w + pw] / pool_area;
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads,
    const Dtype* rand_idx, const Dtype* top_diff,
    const Dtype* bottom_size, const int shrink_factor,
    const int num, const int channels, const int height,
    const int width, const int pool_h, const int pool_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w, 
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    //
    int img_height = (bottom_size) ? (bottom_size[2 * n] - 1) / shrink_factor + 1 : height;
    int img_width  = (bottom_size) ? (bottom_size[2 * n + 1] - 1) / shrink_factor + 1: width;
    if (bottom_size) {
      kernel_h = img_height / pool_h;
      kernel_w = img_width / pool_w;
      stride_h = kernel_h;
      stride_w = kernel_w;
    }
    int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int phend = min(h / stride_h + 1, pool_h);
    int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    int pwend = min(w / stride_w + 1, pool_w);
    Dtype gradient = 0;
    rand_idx += (n * channels + c) * pool_h * pool_w;
    top_diff += (n * channels + c) * pool_h * pool_w;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff[ph * pool_w + pw] *
            (index == static_cast<int>(rand_idx[ph * pool_w + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (do_spm_ && propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to size inputs.";
  }
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* bottom_size = (this->layer_param_.pooling_param().do_spm()) ?
    bottom[1]->gpu_data() : 0;
  const int count = bottom[0]->count();
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
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	count, top_diff, bottom_size, shrink_factor_, mask, top_mask, top[0]->num(), channels_,
        height_, width_, pool_h_, pool_w_,
        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	count, top_diff, bottom_size, shrink_factor_, top[0]->num(), channels_,
        height_, width_, pool_h_, pool_w_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff, bottom_size, shrink_factor_,
        top[0]->num(), channels_, height_, width_, pool_h_,
        pool_w_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);


}  // namespace caffe
