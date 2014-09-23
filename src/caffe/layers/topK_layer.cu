#include <algorithm>
#include <limits>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype>
__global__ void TopKForward(const int n, const int single_count,
                            const Dtype* in_data,
                            const size_t* thresholds,
                            unsigned int* mask,
                            Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, n) {
    const int thresh_idx = index % single_count;
    size_t th = thresholds[thresh_idx];
    mask[index] = in_data[index] > th ? static_cast<uint>(1) : static_cast<uint>(0);
    out_data[index] = in_data[index] * mask[index];
  }
}

template <typename Dtype>
void TopKLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  uint* mask = mask_.mutable_gpu_data();
  const int num = bottom[0]->num();

  const int single_count = bottom[0]->count() / bottom[0]->num();
  const int count = bottom[0]->count();

  caffe_gpu_set(count, Dtype(0), top_data);
  CUDA_CHECK(cudaMemset(mask, 0, count));  // NOLINT(caffe/alt_fn)
  thrust::device_vector<size_t> thresholds(num);

  for (int index=0; index < num; ++index)
    {
      const size_t offset = single_count*index;
      thrust::device_vector<Dtype> values(single_count);
      for (size_t i = 0; i < single_count; ++i) {
          values[i] = bottom_data[i+offset];
        }
      thrust::sort(values.begin(),values.end());
      thresholds[index] = static_cast<size_t>(values[single_count - uint_k_ -1 ]);
    }

  // NOLINT_NEXT_LINE(whitespace/operators)
  TopKForward<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS>>>(
    count, single_count, bottom_data,
    thrust::raw_pointer_cast(&thresholds[0]), mask, top_data);
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void TopKBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index]  * mask[index];
  }
}

template <typename Dtype>
void TopKLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const uint* mask = mask_.gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    TopKBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_CLASS(TopKLayer);


}  // namespace caffe
