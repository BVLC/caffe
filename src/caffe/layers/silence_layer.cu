#include <vector>

#include "caffe/layers/silence_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

template<typename Dtype>
void SilenceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  // Do nothing.
}

template<typename Dtype>
void SilenceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom) {
  for (int_tp i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
        caffe_gpu_set(bottom[i]->count(), Dtype(0),
                      bottom[i]->mutable_gpu_diff());
#endif  // USE_CUDA
      } else {
#ifdef USE_GREENTEA
        viennacl::ocl::context &ctx = viennacl::ocl::get_context(
            this->device_->id());
        viennacl::ocl::program &program = this->device_->program();

        viennacl::ocl::kernel &oclk_gpu_set = program.get_kernel(
            CL_KERNEL_SELECT("gpu_set"));
        viennacl::ocl::enqueue(
            oclk_gpu_set(
                bottom[i]->count(), Dtype(0),
                WrapHandle((cl_mem) bottom[i]->mutable_gpu_diff(), &ctx)),
            ctx.get_queue());
        ctx.get_queue().finish();
#endif
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SilenceLayer);

}  // namespace caffe
