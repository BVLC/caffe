#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void SigmoidForward(const int_tp n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = 1. / (1. + exp(-in[index]));
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void SigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int_tp count = bottom[0]->count();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // NOLINT_NEXT_LINE(whitespace/operators)
    SigmoidForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                                      CAFFE_CUDA_NUM_THREADS)(
        count, bottom_data, top_data);
    CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    viennacl::ocl::kernel &oclk_sigmoid = program.get_kernel(
        CL_KERNEL_SELECT("sigmoid_forward"));
    viennacl::ocl::enqueue(
        oclk_sigmoid(count, WrapHandle((cl_mem) bottom_data, &ctx),
                     WrapHandle((cl_mem) top_data, &ctx)),
        ctx.get_queue());
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
__global__ void SigmoidBackward(const int_tp n, const Dtype* in_diff,
                                const Dtype* out_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype sigmoid_x = out_data[index];
    out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void SigmoidLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int_tp count = bottom[0]->count();

    if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
      // NOLINT_NEXT_LINE(whitespace/operators)
      SigmoidBackward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                                         CAFFE_CUDA_NUM_THREADS)(
          count, top_diff, top_data, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(
          this->device_->id());
      viennacl::ocl::program &program = this->device_->program();

      viennacl::ocl::kernel &oclk_sigmoid = program.get_kernel(
          CL_KERNEL_SELECT("sigmoid_backward"));
      viennacl::ocl::enqueue(
          oclk_sigmoid(count, WrapHandle((cl_mem) top_diff, &ctx),
                       WrapHandle((cl_mem) top_data, &ctx),
                       WrapHandle((cl_mem) bottom_diff, &ctx)),
          ctx.get_queue());
#endif  // USE_GREENTEA
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidLayer);

}  // namespace caffe
