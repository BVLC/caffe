#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

template<typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL)<< this->type()
    << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int_tp count = bottom[0]->count();
    const int_tp num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
      // First, compute the diff
      caffe_copy(count, sigmoid_output_data, bottom_diff);
      caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
      // Scale down gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0];
      caffe_gpu_scal(count, loss_weight / num, bottom_diff);
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(
          this->device_->id());

      // First, compute the diff
      greentea_copy<Dtype>(count, (cl_mem)sigmoid_output_data, 0,
                           (cl_mem)bottom_diff, 0, &ctx);
      greentea_gpu_axpy<Dtype>(this->device_->id(), count,
                               Dtype(-1), (cl_mem)target, 0,
                               (cl_mem)bottom_diff, 0);
      // Scale down gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0];
      greentea_gpu_scal(this->device_->id(), count, loss_weight / num,
                        (cl_mem)bottom_diff, 0);
#endif  // USE_GREENTEA
    }
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(SigmoidCrossEntropyLossLayer);

}  // namespace caffe
