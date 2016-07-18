#include <algorithm>
#include <cfloat>
#include <vector>

#ifdef USE_CUDA
#include "thrust/device_vector.h"
#endif

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/util/im2col.hpp"
#endif

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void kernel_channel_max(const int_tp num, const int_tp channels,
                                   const int_tp spatial_dim, const Dtype* data,
                                   Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int_tp n = index / spatial_dim;
    int_tp s = index % spatial_dim;
    Dtype maxval = -FLT_MAX;
    for (int_tp c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template<typename Dtype>
__global__ void kernel_channel_subtract(const int_tp count, const int_tp num,
                                        const int_tp channels,
                                        const int_tp spatial_dim,
                                        const Dtype* channel_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int_tp n = index / channels / spatial_dim;
    int_tp s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

template<typename Dtype>
__global__ void kernel_exp(const int_tp count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}

template<typename Dtype>
__global__ void kernel_channel_sum(const int_tp num, const int_tp channels,
                                   const int_tp spatial_dim, const Dtype* data,
                                   Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int_tp n = index / spatial_dim;
    int_tp s = index % spatial_dim;
    Dtype sum = 0;
    for (int_tp c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template<typename Dtype>
__global__ void kernel_channel_div(const int_tp count, const int_tp num,
                                   const int_tp channels,
                                   const int_tp spatial_dim,
                                   const Dtype* channel_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int_tp n = index / channels / spatial_dim;
    int_tp s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

template<typename Dtype>
__global__ void kernel_channel_dot(const int_tp num, const int_tp channels,
                                   const int_tp spatial_dim,
                                   const Dtype* data_1, const Dtype* data_2,
                                   Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int_tp n = index / spatial_dim;
    int_tp s = index % spatial_dim;
    Dtype dot = 0;
    for (int_tp c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}
#endif

template<typename Dtype>
void SoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int_tp count = bottom[0]->count();
  int_tp channels = top[0]->shape(softmax_axis_);

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // CUDA backend code
    caffe_copy(count, bottom_data, top_data);
    // We need to subtract the max to avoid numerical issues, compute the exp,
    // and then normalize.
    // compute max
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_max<Dtype> CUDA_KERNEL(
        CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
        CAFFE_CUDA_NUM_THREADS)(outer_num_, channels, inner_num_, top_data,
        scale_data);
    // subtract
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_subtract<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS)(count, outer_num_, channels, inner_num_,
        scale_data, top_data);
    // exponentiate
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_exp<Dtype> CUDA_KERNEL(
        CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS)(count, top_data,
        top_data);
    // sum after exp
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_sum<Dtype> CUDA_KERNEL(
        CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
        CAFFE_CUDA_NUM_THREADS)(outer_num_, channels,
            inner_num_, top_data, scale_data);
    // divide
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_div<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS)(count, outer_num_, channels, inner_num_,
        scale_data, top_data);
#endif
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    caffe_copy(count, bottom_data, top_data);

    viennacl::ocl::kernel &oclk_channel_max = program.get_kernel(
        CL_KERNEL_SELECT("kernel_channel_max"));

    ClState& clState = Caffe::cl_state();
    ClMemOff<Dtype> buf_top = clState.get_buffer_mem(top_data);
    ClMemOff<Dtype> buf_scale = clState.get_buffer_mem(scale_data);

    viennacl::ocl::enqueue(
        oclk_channel_max(outer_num_, channels, inner_num_,
                         WrapHandle(buf_top.memobj, &ctx),
                         WrapHandle(buf_scale.memobj, &ctx)),
        ctx.get_queue());

    viennacl::ocl::kernel &oclk_channel_subtract = program.get_kernel(
        CL_KERNEL_SELECT("kernel_channel_subtract"));
    viennacl::ocl::enqueue(
        oclk_channel_subtract(count, outer_num_, channels, inner_num_,
                              WrapHandle(buf_scale.memobj, &ctx),
                              WrapHandle(buf_top.memobj, &ctx)),
        ctx.get_queue());

    viennacl::ocl::kernel &oclk_exp = program.get_kernel(
        CL_KERNEL_SELECT("kernel_exp"));
    viennacl::ocl::enqueue(
        oclk_exp(count,
                 WrapHandle(buf_top.memobj, &ctx),
                 WrapHandle(buf_top.memobj, &ctx)),
        ctx.get_queue());

    viennacl::ocl::kernel &oclk_channel_sum = program.get_kernel(
        CL_KERNEL_SELECT("kernel_channel_sum"));
    viennacl::ocl::enqueue(
        oclk_channel_sum(outer_num_, channels, inner_num_,
                         WrapHandle(buf_top.memobj, &ctx),
                         WrapHandle(buf_scale.memobj, &ctx)),
        ctx.get_queue());

    viennacl::ocl::kernel &oclk_channel_div = program.get_kernel(
        CL_KERNEL_SELECT("kernel_channel_div"));
    viennacl::ocl::enqueue(
        oclk_channel_div(count, outer_num_, channels, inner_num_,
                         WrapHandle(buf_scale.memobj, &ctx),
                         WrapHandle(buf_top.memobj, &ctx)),
        ctx.get_queue());
#endif
  }
}

template<typename Dtype>
void SoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int_tp count = top[0]->count();
  int_tp channels = top[0]->shape(softmax_axis_);

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    caffe_copy(top[0]->count(), top_diff, bottom_diff);
    // Compute inner1d(top_diff, top_data) and
    // subtract them from the bottom diff.
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_dot<Dtype> CUDA_KERNEL(
        CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
        CAFFE_CUDA_NUM_THREADS)(outer_num_, channels, inner_num_,
            top_diff, top_data, scale_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_subtract<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS)(count, outer_num_, channels, inner_num_,
        scale_data, bottom_diff);
    // elementwise multiplication
    caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
#endif
  } else {
#ifdef USE_GREENTEA

    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    caffe_copy(top[0]->count(), top_diff, bottom_diff);

    viennacl::ocl::kernel &oclk_channel_dot = program.get_kernel(
        CL_KERNEL_SELECT("kernel_channel_dot"));

    ClState& clState = Caffe::cl_state();
    ClMemOff<Dtype> buf_top_diff = clState.get_buffer_mem(top_diff);
    ClMemOff<Dtype> buf_top_data = clState.get_buffer_mem(top_data);
    ClMemOff<Dtype> buf_scale = clState.get_buffer_mem(scale_data);
    ClMemOff<Dtype> buf_bottom = clState.get_buffer_mem(bottom_diff);

    viennacl::ocl::enqueue(
        oclk_channel_dot(outer_num_, channels, inner_num_,
                         WrapHandle(buf_top_diff.memobj, &ctx),
                         WrapHandle(buf_top_data.memobj, &ctx),
                         WrapHandle(buf_scale.memobj, &ctx)),
        ctx.get_queue());

    viennacl::ocl::kernel &oclk_channel_subtract = program.get_kernel(
        CL_KERNEL_SELECT("kernel_channel_subtract"));
    viennacl::ocl::enqueue(
        oclk_channel_subtract(count, outer_num_, channels, inner_num_,
                              WrapHandle(buf_scale.memobj, &ctx),
                              WrapHandle(buf_bottom.memobj, &ctx)),
        ctx.get_queue());

    caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
#endif
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxLayer);

}  // namespace caffe
