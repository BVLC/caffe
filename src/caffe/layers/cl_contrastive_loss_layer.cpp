#ifdef USE_OCL
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/contrastive_loss_layer.hpp"

extern "C" const char _cl_contrastive_loss_layer_start;
extern "C" const char _cl_contrastive_loss_layer_end;

namespace caffe {

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      diff_.mutable_gpu_data());  // a_i-b_i
  caffe_gpu_powx(
      count,
      diff_.mutable_gpu_data(),  // a_i-b_i
      Dtype(2),
      diff_sq_.mutable_gpu_data());  // (a_i-b_i)^2
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_.gpu_data(),  // (a_i-b_i)^2
      summer_vec_.gpu_data(),
      Dtype(0.0),
      dist_sq_.mutable_gpu_data());  // \Sum (a_i-b_i)^2
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
      if (legacy_version) {
        loss += std::max(margin - dist_sq_.cpu_data()[i], Dtype(0.0));
      } else {
        Dtype dist = std::max(margin - (Dtype)sqrt(dist_sq_.cpu_data()[i]),
                              Dtype(0.0));
        loss += dist*dist;
      }
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  ClState& state = Caffe::cl_state();
  state.submit_program("contrastive_loss",
      &_cl_contrastive_loss_layer_start, &_cl_contrastive_loss_layer_end);
  const bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  ClKernel kernel = state.get_kernel(
      legacy_version ? "CLLBackward" : "CLLBackwardLegacy");

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const int count = bottom[0]->count();
      const int channels = bottom[0]->channels();
      Dtype margin = this->layer_param_.contrastive_loss_param().margin();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[0]->num());

      cl_uint argIdx = 0;
      kernel.set_arg(argIdx++, count);
      kernel.set_arg(argIdx++, channels);
      kernel.set_arg(argIdx++, margin);
      kernel.set_arg(argIdx++, alpha);
      kernel.set_arg(argIdx++, bottom[2]->gpu_data());
      kernel.set_arg(argIdx++, diff_.gpu_data());
      kernel.set_arg(argIdx++, dist_sq_.gpu_data());
      kernel.set_arg(argIdx++, bottom[i]->mutable_gpu_diff());
      kernel.enqueue(count);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ContrastiveLossLayer);

}  // namespace caffe
#endif // USE_OCL
