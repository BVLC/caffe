#ifdef USE_OCL
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

extern "C" const char _cl_softmax_loss_layer_start;
extern "C" const char _cl_softmax_loss_layer_end;

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int size = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();

  ClState& state = Caffe::cl_state();
  state.submit_program("softmaxloss", &_cl_softmax_loss_layer_start,
      &_cl_softmax_loss_layer_end);

  ClKernel kernel = state.get_kernel("SoftmaxLossForward");
  cl_uint argIdx = 0;
  int has_ignore_label_int = has_ignore_label_ ? 1 : 0;
  kernel.set_arg(argIdx++, size);
  kernel.set_arg(argIdx++, prob_data);
  kernel.set_arg(argIdx++, label);
  kernel.set_arg(argIdx++, loss_data);
  kernel.set_arg(argIdx++, outer_num_);
  kernel.set_arg(argIdx++, dim);
  kernel.set_arg(argIdx++, inner_num_);
  kernel.set_arg(argIdx++, has_ignore_label_int);
  kernel.set_arg(argIdx++, ignore_label_);
  kernel.set_arg(argIdx++, counts);
  kernel.enqueue(size);

  Dtype loss;
  caffe_gpu_asum(size, loss_data, &loss);
  if (normalize_) {
    Dtype count;
    caffe_gpu_asum(size, counts, &count);
    loss /= count;
  } else {
    loss /= outer_num_;
  }
  top[0]->mutable_cpu_data()[0] = loss;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int size = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();

    ClState& state = Caffe::cl_state();
    state.submit_program("softmaxloss", &_cl_softmax_loss_layer_start,
        &_cl_softmax_loss_layer_end);

    ClKernel kernel = state.get_kernel("SoftmaxLossBackward");
    cl_uint argIdx = 0;
    int has_ignore_label_int = has_ignore_label_ ? 1 : 0;
    kernel.set_arg(argIdx++, size);
    kernel.set_arg(argIdx++, top_data);
    kernel.set_arg(argIdx++, label);
    kernel.set_arg(argIdx++, bottom_diff);
    kernel.set_arg(argIdx++, outer_num_);
    kernel.set_arg(argIdx++, dim);
    kernel.set_arg(argIdx++, inner_num_);
    kernel.set_arg(argIdx++, has_ignore_label_int);
    kernel.set_arg(argIdx++, ignore_label_);
    kernel.set_arg(argIdx++, counts);
    kernel.enqueue(size);

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      Dtype count;
      caffe_gpu_asum(size, counts, &count);
      caffe_gpu_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);


}  // namespace caffe
#endif  // USE_OCL
