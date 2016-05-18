#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
inline __device__ Dtype
backward_i(const Dtype a, const Dtype s, const Dtype t) {
  return a * (s - t);
}

template <typename Dtype>
__global__ void backward_with_ignore_kernel(const int n,
                                            const Dtype ignore_label,
                                            const Dtype a,
                                            const Dtype* __restrict__ target,
                                            const Dtype* __restrict__ sigmoid,
                                            Dtype* __restrict__ diff) {
  CUDA_KERNEL_LOOP(index, n) {
    diff[index] = (ignore_label != target[index]) *
                  backward_i(a, sigmoid[index], target[index]);
  }
}

template <typename Dtype>
__global__ void backward_kernel(const int n, const Dtype a,
                                const Dtype* __restrict__ target,
                                const Dtype* __restrict__ sigmoid,
                                Dtype* __restrict__ diff) {
  CUDA_KERNEL_LOOP(index, n) {
    diff[index] = backward_i(a, sigmoid[index], target[index]);
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    const Dtype scale = loss_weight / num;

    int grid_size = CAFFE_GET_BLOCKS(count);
    int num_threads = CAFFE_CUDA_NUM_THREADS;

    if (has_ignore_label_) {
      backward_with_ignore_kernel << <grid_size, num_threads>>>
          (count, ignore_label_, scale, target, sigmoid_output_data,
           bottom_diff);
    } else {
      backward_kernel << <grid_size, num_threads>>>
          (count, scale, target, sigmoid_output_data, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(SigmoidCrossEntropyLossLayer);

}  // namespace caffe
