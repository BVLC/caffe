#include "caffe/device.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#ifdef USE_CUDA
template <typename Dtype>
__global__ void AdaDeltaUpdate(int_tp N, Dtype* g, Dtype* h, Dtype* h2,
    Dtype momentum, Dtype delta, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = h[i] = momentum * h[i] + (1-momentum) * gi * gi;
    gi = gi * sqrt((h2[i] + delta) / (hi + delta));
    h2[i] = momentum * h2[i] + (1-momentum) * gi * gi;
    g[i] = local_rate * gi;
  }
}
#endif

template <typename Dtype>
void adadelta_update_gpu(device* dev, int_tp N, Dtype* g, Dtype* h, Dtype* h2,
                         Dtype momentum, Dtype delta, Dtype local_rate) {
  if (dev->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    AdaDeltaUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
    CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS) (
        N, g, h, h2, momentum, delta, local_rate);
    CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev->id());
    viennacl::ocl::program &program = dev->program();
    viennacl::ocl::kernel &oclk_ada_delta_update = program.get_kernel(
        CL_KERNEL_SELECT("ada_delta_update"));
    viennacl::ocl::enqueue(
        oclk_ada_delta_update(N, WrapHandle((cl_mem) g, &ctx),
                              WrapHandle((cl_mem) h, &ctx),
                              WrapHandle((cl_mem) h2, &ctx), momentum, delta,
                              local_rate),
        ctx.get_queue());
#endif  // USE_GREENTEA
  }
}
template void adadelta_update_gpu<float>(device*, int_tp, float*, float*,
                                         float*, float, float, float);
template void adadelta_update_gpu<double>(device*, int_tp, double*, double*,
                                          double*, double, double, double);

}  // namespace caffe
