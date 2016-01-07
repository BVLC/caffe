#include "caffe/device.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void RMSPropUpdate(int_tp N, Dtype* g, Dtype* h, Dtype rms_decay,
                              Dtype delta, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = h[i] = rms_decay * h[i] + (1 - rms_decay) * gi * gi;
    g[i] = local_rate * g[i] / (sqrt(hi) + delta);
  }
}
#endif

template <typename Dtype>
void rmsprop_update_gpu(device* dev, int_tp N, Dtype* g, Dtype* h,
                        Dtype rms_decay, Dtype delta, Dtype local_rate) {
  if (dev->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    RMSPropUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
    CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
        N, g, h, rms_decay, delta, local_rate);
    CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev->id());
    viennacl::ocl::program &program = dev->program();
    viennacl::ocl::kernel &oclk_rms_prop_update = program.get_kernel(
        CL_KERNEL_SELECT("rms_prop_update"));
    viennacl::ocl::enqueue(
        oclk_rms_prop_update(N, WrapHandle((cl_mem) g, &ctx),
                              WrapHandle((cl_mem) h, &ctx),
                              rms_decay, delta,
                              local_rate),
        ctx.get_queue());
#endif  // USE_GREENTEA
  }
}

template void rmsprop_update_gpu<float>(device*, int_tp, float*, float*, float,
                                        float, float);
template void rmsprop_update_gpu<double>(device*, int_tp, double*, double*,
                                         double, double, double);

}  // namespace caffe
