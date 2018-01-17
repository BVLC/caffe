#include <algorithm>
#include <cfloat>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>


#include "caffe/layers/saliency_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SalPoolForward(const int nthreads,
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
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype Features = -FLT_MAX;
    const Dtype* const bottom_slice = image_data + (n * channels + c) * height * width;

    // Saliency map variables
    const Dtype* const saliency_map_slice = saliency_data + (n * channels + c)  * height * width;

    Dtype vmax = -FLT_MAX;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > vmax) {
          Features = bottom_slice[h * width + w] * saliency_map_slice[h * width + w];
          //Si = bottom_slice[h * width + w];
        }
      }
    }
    //printf("Index: %d\tValue: %d\n", index,  (kernel_h * kernel_w));

    top_data[index] =  Features / ( kernel_h * kernel_w );
  }
}


template <typename Dtype>
void SaliencyPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top){
    const Dtype* image_data = bottom[0]->gpu_data();
    const Dtype* saliency_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int count = top[0]->count();
/*
    printf("-------------------------------------------------\n");
    printf("Bottom[0]: Image\n");
    printf("Num: \t%d\n", bottom[0]->num());
    printf("Channels: \t%d\n", bottom[0]->channels());
    printf("Height: \t%d\n", bottom[0]->height());
    printf("Width: \t%d\n", bottom[0]->width());
    printf("\nBottom[1]: Saliency Map\n");
    printf("Num: \t%d\n", bottom[1]->num());
    printf("Channels: \t%d\n", bottom[1]->channels());
    printf("Height: \t%d\n", bottom[1]->height());
    printf("Width: \t%d\n\n", bottom[1]->width());
    printf("Count:\t %d\n", bottom[0]->count());
    printf("-------------------------------------------------\n");
*/
    // NOLINT_NEXT_LINE(whitespace/operators)
    SalPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, image_data, saliency_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
    CUDA_POST_KERNEL_CHECK;
  }


template <typename Dtype>
__global__ void SalPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
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
      kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
      bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SaliencyPoolingLayer);

}  // namespace caffe
