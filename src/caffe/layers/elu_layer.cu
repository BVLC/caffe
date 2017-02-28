#include <algorithm>
#include <vector>

#include "caffe/layers/elu_layer.hpp"

namespace caffe {

#ifdef USE_CUDA
template <typename Dtype>
__global__ void ELUForward(const int n, const Dtype* in, Dtype* out,
    Dtype alpha) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] :
        alpha * (exp(in[index]) - 1);
  }
}
#endif  // USE_CUDA

template <typename Dtype>
void ELULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype alpha = this->layer_param_.elu_param().alpha();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // NOLINT_NEXT_LINE(whitespace/operators)
    ELUForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS)(
        count, bottom_data, top_data, alpha);
    CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    viennacl::ocl::kernel &oclk_elu = program.get_kernel(
        CL_KERNEL_SELECT("elu_forward"));
    viennacl::ocl::enqueue(
        oclk_elu(count, WrapHandle((cl_mem) bottom_data, &ctx),
                  WrapHandle((cl_mem) top_data, &ctx), alpha),
        ctx.get_queue());
#endif  // USE_GREENTEA
  }
}

#ifdef USE_CUDA
template <typename Dtype>
__global__ void ELUBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, const Dtype* in_data,
    Dtype* out_diff, Dtype alpha) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_data[index] > 0 ? in_diff[index] :
        in_diff[index] * (out_data[index] + alpha);
  }
}
#endif  // USE_CUDA

template <typename Dtype>
void ELULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype alpha = this->layer_param_.elu_param().alpha();

    if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
      // NOLINT_NEXT_LINE(whitespace/operators)
      ELUBackward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                                     CAFFE_CUDA_NUM_THREADS)(
          count, top_diff, top_data, bottom_data, bottom_diff, alpha);
      CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(
          this->device_->id());
      viennacl::ocl::program &program = this->device_->program();

      viennacl::ocl::kernel &oclk_elu = program.get_kernel(
          CL_KERNEL_SELECT("elu_backward"));
      viennacl::ocl::enqueue(
          oclk_elu(count, WrapHandle((cl_mem) top_diff, &ctx),
                   WrapHandle((cl_mem) top_data, &ctx),
                   WrapHandle((cl_mem) bottom_data, &ctx),
                   WrapHandle((cl_mem) bottom_diff, &ctx), alpha),
          ctx.get_queue());
#endif  // USE_GREENTEA
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ELULayer);


}  // namespace caffe
