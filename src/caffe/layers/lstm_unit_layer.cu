#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/lstm_layer.hpp"

namespace caffe {

#ifdef USE_CUDA
template <typename Dtype>
__device__ Dtype sigmoid(const Dtype x) {
  return Dtype(1) / (Dtype(1) + exp(-x));
}

template <typename Dtype>
__device__ Dtype tanh(const Dtype x) {
  return Dtype(2) * sigmoid(Dtype(2) * x) - Dtype(1);
}

template <typename Dtype>
__global__ void LSTMActsForward(const int_tp nthreads, const int_tp dim,
                                const Dtype* X, Dtype* X_acts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int_tp x_dim = 4 * dim;
    const int_tp d = index % x_dim;
    if (d < 3 * dim) {
      X_acts[index] = sigmoid(X[index]);
    } else {
      X_acts[index] = tanh(X[index]);
    }
  }
}

template <typename Dtype>
__global__ void LSTMUnitForward(const int_tp nthreads, const int_tp dim,
    const Dtype* C_prev, const Dtype* X, const Dtype* cont,
    Dtype* C, Dtype* H) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int_tp n = index / dim;
    const int_tp d = index % dim;
    const Dtype* X_offset = X + 4 * dim * n;
    const Dtype i = X_offset[d];
    const Dtype f = X_offset[1 * dim + d];
    const Dtype o = X_offset[2 * dim + d];
    const Dtype g = X_offset[3 * dim + d];
    const Dtype c_prev = C_prev[index];
    const Dtype c = cont[n] * f * c_prev + i * g;
    C[index] = c;
    const Dtype tanh_c = tanh(c);
    H[index] = o * tanh_c;
  }
}
#endif  // USE_CUDA

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int_tp count = top[1]->count();
  const Dtype* C_prev = bottom[0]->gpu_data();
  const Dtype* X = bottom[1]->gpu_data();
  const Dtype* cont = bottom[2]->gpu_data();
  Dtype* X_acts = X_acts_.mutable_gpu_data();
  Dtype* C = top[0]->mutable_gpu_data();
  Dtype* H = top[1]->mutable_gpu_data();
  const int_tp X_count = bottom[1]->count();
  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // NOLINT_NEXT_LINE(whitespace/operators)
    LSTMActsForward<Dtype>CUDA_KERNEL(CAFFE_GET_BLOCKS(X_count),
                                      CAFFE_CUDA_NUM_THREADS)(
        X_count, hidden_dim_, X, X_acts);
    CUDA_POST_KERNEL_CHECK;
    // NOLINT_NEXT_LINE(whitespace/operators)
    LSTMUnitForward<Dtype>CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                                      CAFFE_CUDA_NUM_THREADS)(
        count, hidden_dim_, C_prev, X_acts, cont, C, H);
    CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();
    viennacl::ocl::kernel &oclk_lstm_acts_forward = program.get_kernel(
        CL_KERNEL_SELECT("lstm_acts_forward"));
    viennacl::ocl::kernel &oclk_lstm_unit_forward = program.get_kernel(
        CL_KERNEL_SELECT("lstm_unit_forward"));

    viennacl::ocl::enqueue(
        oclk_lstm_acts_forward(X_count, hidden_dim_,
          WrapHandle((cl_mem)X, &ctx),
          WrapHandle((cl_mem)X_acts, &ctx)),
        ctx.get_queue());
    viennacl::ocl::enqueue(
        oclk_lstm_unit_forward(count, hidden_dim_,
          WrapHandle((cl_mem)C_prev, &ctx),
          WrapHandle((cl_mem)X_acts, &ctx),
          WrapHandle((cl_mem)cont, &ctx),
          WrapHandle((cl_mem)C, &ctx),
          WrapHandle((cl_mem)H, &ctx)),
        ctx.get_queue());
#endif  // USE_GREENTEA
  }
}

#ifdef USE_CUDA
template <typename Dtype>
__global__ void LSTMUnitBackward(const int_tp nthreads, const int_tp dim,
    const Dtype* C_prev, const Dtype* X, const Dtype* C, const Dtype* H,
    const Dtype* cont, const Dtype* C_diff, const Dtype* H_diff,
    Dtype* C_prev_diff, Dtype* X_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int_tp n = index / dim;
    const int_tp d = index % dim;
    const Dtype* X_offset = X + 4 * dim * n;
    const Dtype i = X_offset[d];
    const Dtype f = X_offset[1 * dim + d];
    const Dtype o = X_offset[2 * dim + d];
    const Dtype g = X_offset[3 * dim + d];
    const Dtype c_prev = C_prev[index];
    const Dtype c = C[index];
    const Dtype tanh_c = tanh(c);
    Dtype* c_prev_diff = C_prev_diff + index;
    Dtype* X_diff_offset = X_diff + 4 * dim * n;
    Dtype* i_diff = X_diff_offset + d;
    Dtype* f_diff = X_diff_offset + 1 * dim + d;
    Dtype* o_diff = X_diff_offset + 2 * dim + d;
    Dtype* g_diff = X_diff_offset + 3 * dim + d;
    const Dtype c_term_diff =
        C_diff[index] + H_diff[index] * o * (1 - tanh_c * tanh_c);
    const Dtype cont_n = cont[n];
    *c_prev_diff = cont_n * c_term_diff * f;
    *i_diff = c_term_diff * g;
    *f_diff = cont_n * c_term_diff * c_prev;
    *o_diff = H_diff[index] * tanh_c;
    *g_diff = c_term_diff * i;
  }
}

template <typename Dtype>
__global__ void LSTMActsBackward(const int_tp nthreads, const int_tp dim,
    const Dtype* X_acts, const Dtype* X_acts_diff, Dtype* X_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int_tp x_dim = 4 * dim;
    const int_tp d = index % x_dim;
    const Dtype X_act = X_acts[index];
    if (d < 3 * dim) {
      X_diff[index] = X_acts_diff[index] * X_act * (Dtype(1) - X_act);
    } else {
      X_diff[index] = X_acts_diff[index] * (Dtype(1) - X_act * X_act);
    }
  }
}
#endif  // USE_CUDA

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[2]) << "Cannot backpropagate to sequence indicators.";
  if (!propagate_down[0] && !propagate_down[1]) { return; }

  const int_tp count = top[1]->count();
  const Dtype* C_prev = bottom[0]->gpu_data();
  const Dtype* X_acts = X_acts_.gpu_data();
  const Dtype* cont = bottom[2]->gpu_data();
  const Dtype* C = top[0]->gpu_data();
  const Dtype* H = top[1]->gpu_data();
  const Dtype* C_diff = top[0]->gpu_diff();
  const Dtype* H_diff = top[1]->gpu_diff();
  Dtype* C_prev_diff = bottom[0]->mutable_gpu_diff();
  Dtype* X_acts_diff = X_acts_.mutable_gpu_diff();
  const int_tp X_count = bottom[1]->count();
  Dtype* X_diff = bottom[1]->mutable_gpu_diff();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    LSTMUnitBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                    CAFFE_CUDA_NUM_THREADS)(count, hidden_dim_,
        C_prev, X_acts, C, H, cont, C_diff, H_diff, C_prev_diff, X_acts_diff);
    CUDA_POST_KERNEL_CHECK;

    LSTMActsBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        CUDA_KERNEL(CAFFE_GET_BLOCKS(X_count),
                    CAFFE_CUDA_NUM_THREADS)(
        X_count, hidden_dim_, X_acts, X_acts_diff, X_diff);
    CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();
    viennacl::ocl::kernel &oclk_lstm_unit_backward = program.get_kernel(
        CL_KERNEL_SELECT("lstm_unit_backward"));
    viennacl::ocl::kernel &oclk_lstm_acts_backward = program.get_kernel(
        CL_KERNEL_SELECT("lstm_acts_backward"));

    viennacl::ocl::enqueue(
        oclk_lstm_unit_backward(count, hidden_dim_,
          WrapHandle((cl_mem)C_prev, &ctx), WrapHandle((cl_mem)X_acts, &ctx),
          WrapHandle((cl_mem)C, &ctx), WrapHandle((cl_mem)H, &ctx),
          WrapHandle((cl_mem)cont, &ctx), WrapHandle((cl_mem)C_diff, &ctx),
          WrapHandle((cl_mem)H_diff, &ctx),
          WrapHandle((cl_mem)C_prev_diff, &ctx),
          WrapHandle((cl_mem)X_acts_diff, &ctx)),
        ctx.get_queue());
    viennacl::ocl::enqueue(
        oclk_lstm_acts_backward(X_count, hidden_dim_,
          WrapHandle((cl_mem)X_acts, &ctx),
          WrapHandle((cl_mem)X_acts_diff, &ctx),
          WrapHandle((cl_mem)X_diff, &ctx)),
        ctx.get_queue());
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LSTMUnitLayer);

}  // namespace caffe
