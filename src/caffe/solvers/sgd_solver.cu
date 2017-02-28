#include "caffe/device.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#ifdef USE_CUDA
template <typename Dtype>
__global__ void SGDUpdate(int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    g[i] = h[i] = momentum*h[i] + local_rate*g[i];
  }
}
#endif

template <typename Dtype>
void sgd_update_gpu(device* dev, int_tp N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate) {
  if (dev->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    SGDUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS) (
        N, g, h, momentum, local_rate);
    CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev->id());
    viennacl::ocl::program &program = dev->program();
    viennacl::ocl::kernel &oclk_sgd_update = program.get_kernel(
        CL_KERNEL_SELECT("sgd_update"));
    viennacl::ocl::enqueue(
        oclk_sgd_update(N, WrapHandle((cl_mem) g, &ctx),
                        WrapHandle((cl_mem) h, &ctx), momentum, local_rate),
        ctx.get_queue());
#endif  // USE_GREENTEA
  }
}
template void sgd_update_gpu<float>(device*, int_tp, float*, float*, float,
                                    float);
template void sgd_update_gpu<double>(device*, int_tp, double*, double*, double,
                                     double);

}  // namespace caffe
