#include <vector>

#include "caffe/layers/multilabel_sigmoid_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void MultiLabelSigmoidLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* target, Dtype* loss,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value >= 0) {
      loss[i] = input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] *
                    (input_data[i] >= 0)));
      counts[i] = 1;
    } else {
      counts[i] = 0;
      loss[i] = 0;
    }
  }
}

 template <typename Dtype>
   __global__ void MultiLabelSigmoidLossIgnoreDiffGPU(const int count,
                                                      const Dtype* target, Dtype* diff, Dtype* counts) {
   CUDA_KERNEL_LOOP(i, count) {
     const int target_value = static_cast<int>(target[i]);
     if (target_value < 0) {
       diff[i] = 0;
       counts[i] = 0;
     } else
       counts[i] = 1;
   }
 }




template <typename Dtype>
void MultiLabelSigmoidLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  Dtype* count_data = bottom[1]->mutable_gpu_diff();
  Dtype valid_count;
  // NOLINT_NEXT_LINE(whitespace/operators)
  MultiLabelSigmoidLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, input_data, target, loss_data,
      count_data);
  caffe_gpu_asum(count, count_data, &valid_count);
  Dtype loss;
  caffe_gpu_asum(count, loss_data, &loss);
  // CPU version divides loss by num, which is the total number of classes,
  // here we divide by the number of classes that are not "dontcare", as in sigmoidcrossentropyloss
  top[0]->mutable_cpu_data()[0] = loss / valid_count;
}

template <typename Dtype>
void MultiLabelSigmoidLossLayer<Dtype>::Backward_gpu(
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
    Dtype* count_data = bottom[1]->mutable_gpu_diff();

    Dtype valid_count;
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
    // Zero out gradient of ignored targets.
    // NOLINT_NEXT_LINE(whitespace/operators)
    MultiLabelSigmoidLossIgnoreDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, target, bottom_diff,count_data);
    caffe_gpu_asum(count, count_data, &valid_count);
    // Scale down gradient
    Dtype loss_weight = top[0]->cpu_diff()[0];
    // here we should divide by valid_count for consistency
    if (valid_count)
      caffe_gpu_scal(count, loss_weight / valid_count, bottom_diff);
    else
      caffe_gpu_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiLabelSigmoidLossLayer);

}  // namespace caffe
