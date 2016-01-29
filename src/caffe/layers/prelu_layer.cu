#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#ifdef USE_CUDA
// CUDA kernele for forward
template<typename Dtype>
__global__ void PReLUForward(const int_tp n, const int_tp channels,
                             const int_tp dim, const Dtype* in, Dtype* out,
                             const Dtype* slope_data, const int_tp div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int_tp c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}

// CUDA kernel for bottom backward
template<typename Dtype>
__global__ void PReLUBackward(const int_tp n, const int_tp channels,
                              const int_tp dim, const Dtype* in_diff,
                              const Dtype* in_data, Dtype* out_diff,
                              const Dtype* slope_data,
                              const int_tp div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int_tp c = (index / dim) % channels / div_factor;
    out_diff[index] = in_diff[index]
        * ((in_data[index] > 0) + (in_data[index] <= 0) * slope_data[c]);
  }
}

// CUDA kernel for element-wise parameter backward
template<typename Dtype>
__global__ void PReLUParamBackward(const int_tp n, const int_tp rows,
                                   const int_tp rowPitch, const Dtype* in_diff,
                                   const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * in_data[index] * (in_data[index] <= 0);
    for (int k = 1; k < rows; k++) {
      out_diff[index] += in_diff[index + k * rowPitch]
          * in_data[index + k * rowPitch]
          * (in_data[index + k * rowPitch] <= 0);
    }
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void PReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int_tp count = bottom[0]->count();
  const int_tp dim = bottom[0]->count(2);
  const int_tp channels = bottom[0]->channels();
  const Dtype* slope_data = this->blobs_[0]->gpu_data();
  const int_tp div_factor = channel_shared_ ? channels : 1;

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // For in-place computation
    if (top[0] == bottom[0]) {
      caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
    }

    // NOLINT_NEXT_LINE(whitespace/operators)
    PReLUForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                                    CAFFE_CUDA_NUM_THREADS)(
        count, channels, dim, bottom_data, top_data, slope_data, div_factor);
    CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    if (top[0] == bottom[0]) {
      greentea_copy<Dtype>(count, (cl_mem) bottom_data, 0,
                           (cl_mem) (bottom_memory_.mutable_gpu_data()), 0,
                           &ctx);
    }

    viennacl::ocl::kernel &oclk_prelu = program.get_kernel(
        CL_KERNEL_SELECT("prelu_forward"));
    viennacl::ocl::enqueue(
        oclk_prelu(count, channels, dim, WrapHandle((cl_mem) bottom_data, &ctx),
                   WrapHandle((cl_mem) top_data, &ctx),
                   WrapHandle((cl_mem) slope_data, &ctx), div_factor),
        ctx.get_queue());

#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void PReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                     const vector<bool>& propagate_down,
                                     const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int_tp count = bottom[0]->count();
  const int_tp dim = bottom[0]->count(2);
  const int_tp channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // Propagate to param
    // Since to write bottom diff will affect top diff if top and bottom blobs
    // are identical (in-place computaion), we first compute param backward to
    // keep top_diff unchanged.
    if (this->param_propagate_down_[0]) {
      Dtype* slope_diff = this->blobs_[0]->mutable_gpu_diff();
      int_tp cdim = channels * dim;

      // compute element-wise diff
      // NOLINT_NEXT_LINE(whitespace/operators)
      PReLUParamBackward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(cdim),
        CAFFE_CUDA_NUM_THREADS)(
        cdim, bottom[0]->num(), top[0]->offset(1), top_diff ,
        bottom_data ,
        backward_buff_.mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
      if (channel_shared_) {
        Dtype dsum;
        caffe_gpu_dot<Dtype>(channels * dim, backward_buff_.gpu_diff(),
         multiplier_.gpu_data(), &dsum);
        caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum), slope_diff);
      } else {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
          backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
          slope_diff);
      }
    }
    // Propagate to bottom
    if (propagate_down[0]) {
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const Dtype* slope_data = this->blobs_[0]->gpu_data();
      int_tp div_factor = channel_shared_ ? channels : 1;
      // NOLINT_NEXT_LINE(whitespace/operators)
      PReLUBackward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS)(
          count, channels, dim, top_diff, bottom_data, bottom_diff, slope_data,
          div_factor);
      CUDA_POST_KERNEL_CHECK;
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    // Propagate to param
    // Since to write bottom diff will affect top diff if top and bottom blobs
    // are identical (in-place computaion), we first compute param backward to
    // keep top_diff unchanged.
    if (this->param_propagate_down_[0]) {
      Dtype* slope_diff = this->blobs_[0]->mutable_gpu_diff();
      int_tp cdim = channels * dim;

      // compute element-wise diff

      viennacl::ocl::kernel &oclk_prelu = program.get_kernel(
          CL_KERNEL_SELECT("prelu_param_backward"));
      viennacl::ocl::enqueue(
          oclk_prelu(cdim, bottom[0]->num(), top[0]->offset(1),
                     WrapHandle((cl_mem)top_diff, &ctx),
              WrapHandle((cl_mem) bottom_data, &ctx),
              WrapHandle((cl_mem) (backward_buff_.mutable_gpu_diff()), &ctx)),
          ctx.get_queue());

      if (channel_shared_) {
        Dtype dsum;
        greentea_gpu_dot<Dtype>(this->device_->id(), channels * dim,
                                (cl_mem) (backward_buff_.gpu_diff()), 0,
                                (cl_mem) (multiplier_.gpu_data()), 0, &dsum);
        greentea_gpu_add_scalar<Dtype>(this->device_->id(),
                                       this->blobs_[0]->count(), Dtype(dsum),
                                       (cl_mem) slope_diff, 0);
      } else {
        greentea_gpu_gemv<Dtype>(this->device_->id(), CblasNoTrans, channels,
                                 dim, 1., (cl_mem) (backward_buff_.gpu_diff()),
                                 0, (cl_mem) (multiplier_.gpu_data()), 0, 1.,
                                 (cl_mem) slope_diff, 0);
      }
    }
    // Propagate to bottom
    if (propagate_down[0]) {
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const Dtype* slope_data = this->blobs_[0]->gpu_data();
      int_tp div_factor = channel_shared_ ? channels : 1;
      viennacl::ocl::kernel &oclk_prelu = program.get_kernel(
          CL_KERNEL_SELECT("prelu_backward"));
      viennacl::ocl::enqueue(
          oclk_prelu(count, channels, dim, WrapHandle((cl_mem) top_diff, &ctx),
                     WrapHandle((cl_mem) bottom_data, &ctx),
                     WrapHandle((cl_mem) bottom_diff, &ctx),
                     WrapHandle((cl_mem) slope_data, &ctx), div_factor),
          ctx.get_queue());
    }
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PReLULayer);

}  // namespace caffe
