#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void ReLUForward(const int_tp n, const Dtype* in, Dtype* out,
                            Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int_tp count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLUForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS)(
        count, bottom_data, top_data, negative_slope);
    CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();
    viennacl::ocl::kernel &oclk_relu_forward = program.get_kernel(
        CL_KERNEL_SELECT("relu_forward"));
    viennacl::ocl::enqueue(
        oclk_relu_forward(count, WrapHandle((cl_mem) bottom_data, &ctx),
                          WrapHandle((cl_mem) top_data, &ctx), negative_slope),
        ctx.get_queue());
    ctx.get_queue().finish();
#endif  // USE_GREENTEA
  }
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

#ifdef USE_CUDA
template<typename Dtype>
__global__ void ReLUBackward(const int_tp n, const Dtype* in_diff,
                             const Dtype* in_data, Dtype* out_diff,
                             Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index]
        * ((in_data[index] > 0) + (in_data[index] <= 0) * negative_slope);
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int_tp count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
      // NOLINT_NEXT_LINE(whitespace/operators)
      ReLUBackward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                                      CAFFE_CUDA_NUM_THREADS)(
          count, top_diff, bottom_data, bottom_diff, negative_slope);
      CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(
          this->device_->id());
      viennacl::ocl::program &program = this->device_->program();
      viennacl::ocl::kernel &oclk_relu_backward = program.get_kernel(
          CL_KERNEL_SELECT("relu_backward"));
      viennacl::ocl::enqueue(
          oclk_relu_backward(count, WrapHandle((cl_mem) top_diff, &ctx),
                             WrapHandle((cl_mem) bottom_data, &ctx),
                             WrapHandle((cl_mem) bottom_diff, &ctx),
                             negative_slope),
          ctx.get_queue());
      ctx.get_queue().finish();
#endif  // USE_GREENTEA
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);

}  // namespace caffe
