#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void EuclideanLossForwardGPU(const int n,
          const Dtype* label_data_, Dtype* diff__data_,
          const int ignore_label_) {
  CUDA_KERNEL_LOOP(index, n) {
    const float DELTA_RANGE = 0.000000000001;
    float ignore_label_low_range =
          static_cast<float>(ignore_label_) - DELTA_RANGE;
    float ignore_label_high_range =
          static_cast<float>(ignore_label_) + DELTA_RANGE;
    if (label_data_[index] > ignore_label_low_range &&
        label_data_[index] < ignore_label_high_range) {
      diff__data_[index] = 0;
    }
  }
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  if (has_ignore_label_) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    EuclideanLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, bottom[1]->gpu_data(),
                                  diff_.mutable_gpu_data(),
                                  ignore_label_);
  }
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
