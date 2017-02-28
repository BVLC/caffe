#include <vector>

#include "caffe/layers/threshold_layer.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void ThresholdForward(const int_tp n, const Dtype threshold,
                                 const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > threshold ? 1 : 0;
  }
}
#endif

template<typename Dtype>
void ThresholdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int_tp count = bottom[0]->count();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // NOLINT_NEXT_LINE(whitespace/operators)
    ThresholdForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                                        CAFFE_CUDA_NUM_THREADS)(
        count, threshold_, bottom_data, top_data);
    CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    viennacl::ocl::kernel &oclk_threshold = program.get_kernel(
        CL_KERNEL_SELECT("threshold"));
    viennacl::ocl::enqueue(
        oclk_threshold(count, threshold_,
                       WrapHandle((cl_mem) bottom_data, &ctx),
                       WrapHandle((cl_mem) top_data, &ctx)),
        ctx.get_queue());
    ctx.get_queue().finish();
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FORWARD(ThresholdLayer);

}  // namespace caffe
