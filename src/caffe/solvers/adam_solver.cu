#include "caffe/device.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#ifdef USE_CUDA
template <typename Dtype>
__global__ void AdamUpdate(int N, Dtype* g, Dtype* m, Dtype* v,
    Dtype beta1, Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float mi = m[i] = m[i]*beta1 + gi*(1-beta1);
    float vi = v[i] = v[i]*beta2 + gi*gi*(1-beta2);
    g[i] = corrected_local_rate * mi / (sqrt(vi) + eps_hat);
  }
}
#endif


template<typename Dtype>
void adam_update_gpu(device* dev, int_tp N, Dtype* g, Dtype* m, Dtype* v,
                     Dtype beta1, Dtype beta2, Dtype eps_hat,
                     Dtype corrected_local_rate) {
  if (dev->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    AdamUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
    CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS) (
        N, g, m, v, beta1, beta2, eps_hat, corrected_local_rate);
    CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev->id());
    viennacl::ocl::program &program = dev->program();
    viennacl::ocl::kernel &oclk_adam_update = program.get_kernel(
        CL_KERNEL_SELECT("adam_update"));
    viennacl::ocl::enqueue(
        oclk_adam_update(N, WrapHandle((cl_mem) g, &ctx),
                         WrapHandle((cl_mem) m, &ctx),
                         WrapHandle((cl_mem) v, &ctx), beta1, beta2, eps_hat,
                         corrected_local_rate),
        ctx.get_queue());
#endif  // USE_GREENTEA
  }
}


template void adam_update_gpu<float>(device*, int_tp, float*, float*, float*,
                                     float, float, float, float);
template void adam_update_gpu<double>(device*, int_tp, double*, double*,
                                      double*, double, double, double, double);

}  // namespace caffe
