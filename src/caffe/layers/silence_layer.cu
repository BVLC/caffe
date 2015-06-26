#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
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
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      if (this->device_context_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
        caffe_gpu_set(bottom[i]->count(), Dtype(0),
                      bottom[i]->mutable_gpu_data());
#endif  // USE_CUDA
      } else {
#ifdef USE_GREENTEA
        viennacl::ocl::context &ctx = viennacl::ocl::get_context(
            this->device_context_->id());
        viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(
            this->device_context_->id());
        viennacl::ocl::kernel &oclk_gpu_set = program.get_kernel(
            CL_KERNEL_SELECT("gpu_set"));
        viennacl::ocl::enqueue(
            oclk_gpu_set(
                bottom[i]->count(), Dtype(0),
                WrapHandle((cl_mem) bottom[i]->mutable_gpu_data(), &ctx)),
            ctx.get_queue());
        ctx.get_queue().finish();
#endif
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SilenceLayer);

}  // namespace caffe
