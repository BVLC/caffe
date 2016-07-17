#include <algorithm>
#include <vector>

#include "caffe/layers/double_margin_contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DoubleMarginContrastiveLossLayer<Dtype>::Forward_gpu(
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
  Dtype margin_gen =
      this->layer_param_.double_margin_contrastive_loss_param().margin_gen();
  Dtype margin_imp =
      this->layer_param_.double_margin_contrastive_loss_param().margin_imp();
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      Dtype dist = std::max(sqrt(dist_sq_.cpu_data()[i]) - margin_gen,
                            Dtype(0.0));
      loss += dist*dist;
    } else {  // dissimilar pairs
      Dtype dist = std::max(margin_imp - sqrt(dist_sq_.cpu_data()[i]),
                            Dtype(0.0));
      loss += dist*dist;
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin_gen, const Dtype margin_imp, const Dtype alpha,
    const Dtype* y, const Dtype* diff, const Dtype* dist_sq,
    Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    Dtype dist = sqrt(dist_sq[n]);
    Dtype mdist(0.0);
    Dtype beta(0.0);
    if (static_cast<int>(y[n])) {  // similar pairs
      mdist = (dist - margin_gen);
      beta = alpha * mdist / (dist + Dtype(1e-4)) * diff[i];
      if (mdist > 0.0) {
        bottom_diff[i] = beta;
      } else {
        bottom_diff[i] = 0;
      }
    } else {  // dissimilar pairs
      mdist = (margin_imp - dist);
      beta = -alpha * mdist / (dist + Dtype(1e-4)) * diff[i];
      if (mdist > 0.0) {
        bottom_diff[i] = beta;
      } else {
        bottom_diff[i] = 0;
      }
    }
  }
}

template <typename Dtype>
void DoubleMarginContrastiveLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const int count = bottom[0]->count();
      const int channels = bottom[0]->channels();
      Dtype margin_gen =
        this->layer_param_.double_margin_contrastive_loss_param().margin_gen();
      Dtype margin_imp =
        this->layer_param_.double_margin_contrastive_loss_param().margin_imp();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[0]->num());
      // NOLINT_NEXT_LINE(whitespace/operators)
      CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin_gen, margin_imp, alpha,
          bottom[2]->gpu_data(),  // pair similarity 0 or 1
          diff_.gpu_data(),  // the cached eltwise difference between a and b
          dist_sq_.gpu_data(),  // the cached square distance between a and b
          bottom[i]->mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DoubleMarginContrastiveLossLayer);

}  // namespace caffe
