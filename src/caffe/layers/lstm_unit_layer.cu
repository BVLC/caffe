#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__device__ Dtype cuda_sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
__device__ Dtype cuda_sigmoid_diff(Dtype x) {
  return x * (1. - x);
}

template <typename Dtype>
__device__ Dtype cuda_tanh(Dtype x) {
  Dtype exp2x = exp(2 * x);
  return fabs(x) < Dtype(5) ? ((exp2x - Dtype(1)) / (exp2x + Dtype(1))) :
    (x > 0 ? Dtype(1) : Dtype(-1));
}

template <typename Dtype>
__device__ Dtype cuda_tanh_diff(Dtype x) {
  return (1. - x * x);
}

template <typename Dtype>
__global__ void ForwardCombineGates(
  int n,
  const Dtype* prev_state_data,
  Dtype* input_gates,
  Dtype* forget_gates,
  Dtype* output_gates,
  Dtype* input_values,
  Dtype* next_memory_state,
  Dtype* next_hidden_state) {
  CUDA_KERNEL_LOOP(idx, n) {
    input_gates[idx] = cuda_sigmoid(input_gates[idx]);
    forget_gates[idx] = cuda_sigmoid(forget_gates[idx]);
    output_gates[idx] = cuda_sigmoid(output_gates[idx]);
    input_values[idx] = cuda_tanh(input_values[idx]);

    next_memory_state[idx] = prev_state_data[idx] * forget_gates[idx] +
        input_gates[idx] * input_values[idx];
    next_hidden_state[idx] = next_memory_state[idx] * output_gates[idx];
  }
}

template <typename Dtype>
__global__ void BackwardGates(
  int n,
  const Dtype* input_gates,
  const Dtype* forget_gates,
  const Dtype* output_gates,
  const Dtype* input_values,
  Dtype* input_gates_diff,
  Dtype* forget_gates_diff,
  Dtype* output_gates_diff,
  Dtype* input_values_diff) {
  CUDA_KERNEL_LOOP(idx, n) {
    input_gates_diff[idx] = cuda_sigmoid_diff(input_gates[idx]);
    forget_gates_diff[idx] = cuda_sigmoid_diff(forget_gates[idx]);
    output_gates_diff[idx] = cuda_sigmoid_diff(output_gates[idx]);
    input_values_diff[idx] = cuda_tanh_diff(input_values[idx]);
  }
}

template <typename Dtype>
void LstmUnitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* prev_state_data = bottom[1]->gpu_data();

  const Dtype* input_weight = this->blobs_[0]->gpu_data();
  const Dtype* input_gate_weight = this->blobs_[1]->gpu_data();
  const Dtype* forget_gate_weight = this->blobs_[2]->gpu_data();
  const Dtype* output_gate_weight = this->blobs_[3]->gpu_data();

  Dtype* next_hidden_state = top[0]->mutable_gpu_data();
  Dtype* next_memory_state = top[1]->mutable_gpu_data();

  Dtype* input_gates = input_gates_data_buffer_->mutable_gpu_data();
  Dtype* forget_gates = forget_gates_data_buffer_->mutable_gpu_data();
  Dtype* output_gates = output_gates_data_buffer_->mutable_gpu_data();
  Dtype* input_values = input_values_data_buffer_->mutable_gpu_data();

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
    (Dtype)1., input_data, input_weight,
    (Dtype)0., input_values);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
    (Dtype)1., input_data, input_gate_weight,
    (Dtype)0., input_gates);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
    (Dtype)1., input_data, forget_gate_weight,
    (Dtype)0., forget_gates);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
    (Dtype)1., input_data, output_gate_weight,
    (Dtype)0., output_gates);

  caffe_gpu_set(channels_ * num_, Dtype(0), forget_gates);
  caffe_gpu_sub(channels_ * num_, forget_gates, output_gates, forget_gates);

  const int count = num_ * channels_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  ForwardCombineGates<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
      count,
      prev_state_data,
      input_gates,
      forget_gates,
      output_gates,
      input_values,
      next_memory_state,
      next_hidden_state);

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void LstmUnitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    caffe_gpu_set(bottom[i]->count(), Dtype(0),
        bottom[i]->mutable_gpu_diff());
  }
  for (int i = 0; i < 4; ++i) {
    caffe_gpu_set(this->blobs_[i]->count(), Dtype(0),
        this->blobs_[i]->mutable_gpu_diff());
  }

  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* prev_state_data = bottom[1]->gpu_data();

  const Dtype* input_weight = this->blobs_[0]->gpu_data();
  const Dtype* input_gate_weight = this->blobs_[1]->gpu_data();
  const Dtype* forget_gate_weight = this->blobs_[2]->gpu_data();
  const Dtype* output_gate_weight = this->blobs_[3]->gpu_data();

  const Dtype* input_gates = input_gates_data_buffer_->gpu_data();
  const Dtype* forget_gates = forget_gates_data_buffer_->gpu_data();
  const Dtype* output_gates = output_gates_data_buffer_->gpu_data();
  const Dtype* input_values = input_values_data_buffer_->gpu_data();

  Dtype* gates_diff = gates_diff_buffer_->mutable_gpu_data();

  Dtype* input_gates_diff = gates_diff + channels_ * num_ * 0;
  Dtype* forget_gates_diff = gates_diff + channels_ * num_ * 1;
  Dtype* output_gates_diff = gates_diff + channels_ * num_ * 2;
  Dtype* input_values_diff = gates_diff + channels_ * num_ * 3;

  const int count = num_ * channels_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  BackwardGates<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count,
    input_gates,
    forget_gates,
    output_gates,
    input_values,
    input_gates_diff,
    forget_gates_diff,
    output_gates_diff,
    input_values_diff);

  CUDA_POST_KERNEL_CHECK;

  Dtype* input_weight_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* input_gate_weight_diff = this->blobs_[1]->mutable_gpu_diff();
  Dtype* forget_gate_weight_diff = this->blobs_[2]->mutable_gpu_diff();
  Dtype* output_gate_weight_diff = this->blobs_[3]->mutable_gpu_diff();

  Dtype* input_diff = bottom[0]->mutable_gpu_diff();
  Dtype* prev_state_diff = bottom[1]->mutable_gpu_diff();

  const Dtype* next_hidden_state_diff = top[0]->gpu_diff();
  const Dtype* next_memory_state = top[1]->gpu_data();
  const Dtype* next_memory_state_diff = top[1]->gpu_diff();

  Dtype* next_state_tot_diff = next_state_tot_diff_buffer_->mutable_gpu_data();
  caffe_gpu_mul(num_ * channels_, output_gates,
    next_hidden_state_diff, next_state_tot_diff);
  caffe_gpu_add(num_ * channels_, next_memory_state_diff,
    next_state_tot_diff, next_state_tot_diff);

  caffe_gpu_mul(num_ * channels_, next_state_tot_diff,
    forget_gates, prev_state_diff);

  Dtype* dldg_data = dldg_buffer_->mutable_gpu_data();

  caffe_gpu_mul(num_ * channels_, input_gates,
    input_values_diff, dldg_data);
  caffe_gpu_mul(num_ * channels_, next_state_tot_diff,
    dldg_data, dldg_data);
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    channels_, input_data_size_, num_,
    (Dtype)1., dldg_data, input_data,
    (Dtype)0., input_weight_diff);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
    num_, input_data_size_, channels_,
    (Dtype)1., dldg_data, input_weight,
    (Dtype)1., input_diff);

  caffe_gpu_mul(num_ * channels_, input_gates_diff,
    input_values, dldg_data);
  caffe_gpu_mul(num_ * channels_, next_state_tot_diff,
    dldg_data, dldg_data);
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    channels_, input_data_size_, num_,
    (Dtype)1., dldg_data, input_data,
    (Dtype)0., input_gate_weight_diff);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
    num_, input_data_size_, channels_,
    (Dtype)1., dldg_data, input_gate_weight,
    (Dtype)1., input_diff);

  caffe_gpu_mul(num_ * channels_, forget_gates_diff,
    prev_state_data, dldg_data);
  caffe_gpu_mul(num_ * channels_, next_state_tot_diff,
    dldg_data, dldg_data);
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    channels_, input_data_size_, num_,
    (Dtype)-1., dldg_data, input_data,
    (Dtype)1., output_gate_weight_diff);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
    num_, input_data_size_, channels_,
    (Dtype)-1., dldg_data, output_gate_weight,
    (Dtype)1., input_diff);

  caffe_gpu_mul(num_ * channels_, output_gates_diff,
    next_memory_state, dldg_data);
  caffe_gpu_mul(num_ * channels_, next_hidden_state_diff,
    dldg_data, dldg_data);
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    channels_, input_data_size_, num_,
    (Dtype)1., dldg_data, input_data,
    (Dtype)1., output_gate_weight_diff);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
    num_, input_data_size_, channels_,
    (Dtype)1., dldg_data, output_gate_weight,
    (Dtype)1., input_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(LstmUnitLayer);

}  // namespace caffe
