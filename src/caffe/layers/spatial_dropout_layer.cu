#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
__global__ void SpatialDropoutForward(const int n, const int channel_size,
    const Dtype* in, const unsigned int* mask, const unsigned int threshold,
    const float scale, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (mask[index/channel_size] > threshold) * scale;
  }
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    const int channel_size = bottom[0]->width() * bottom[0]->height();
    const int rand_count = bottom[0]->num() * bottom[0]->channels();

    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    caffe_gpu_rng_uniform(rand_count, mask);
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    SpatialDropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(
        count, channel_size, bottom_data, mask, uint_thres_, scale_, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void SpatialDropoutBackward(const int n, const int channel_size,
    const Dtype* in_diff, const unsigned int* mask,
    const unsigned int threshold, const float scale, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    int mask_index = index/channel_size;
    out_diff[index] = in_diff[index] * scale * (mask[mask_index] > threshold);
  }
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask =
          static_cast<const unsigned int*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
      const int channel_size =  bottom[0]->width() * bottom[0]->height();
      // NOLINT_NEXT_LINE(whitespace/operators)
      SpatialDropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
          count, channel_size, top_diff, mask, uint_thres_, scale_,
          bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialDropoutLayer);


}  // namespace caffe
