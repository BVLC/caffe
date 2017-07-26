#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/weighted_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void WeightedSoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, const Dtype* weight,
          Dtype* loss, const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    const Dtype weight_value = weight[n * spatial_dim + s];
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -weight_value
        *log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      counts[index] = weight_value;
    }
  }
}

template <typename Dtype>
void WeightedSoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  this->softmax_layer_->Forward(this->softmax_bottom_vec_,
                                this->softmax_top_vec_);
  const Dtype* prob_data = this->prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* weight = bottom[2]->gpu_data();
  const int dim = this->prob_.count() / this->outer_num_;
  const int nthreads = this->outer_num_ * this->inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = this->prob_.mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  WeightedSoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, weight, loss_data,
      this->outer_num_, dim, this->inner_num_, this->has_ignore_label_,
      this->ignore_label_, counts);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (this->normalization_ == LossParameter_NormalizationMode_VALID &&
      this->has_ignore_label_) {
    caffe_gpu_asum(nthreads, counts, &valid_count);
  }
  top[0]->mutable_cpu_data()[0] = loss
    / this->get_normalizer(this->normalization_, valid_count);
  if (top.size() == 2) {
    top[1]->ShareData(this->prob_);
  }
}

template <typename Dtype>
__global__ void WeightedSoftmaxLossBackwardGPU(const int nthreads,
          const Dtype* top, const Dtype* label, const Dtype* weight,
          Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    const Dtype weight_value = weight[n * spatial_dim + s];
    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= weight_value;
      counts[index] = weight_value;
    }
  }
}

template <typename Dtype>
void WeightedSoftmaxWithLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to weight inputs.";
  }
  if (propagate_down[0]) {
    tile_layer_->Forward(tile_bottom_vec_, tile_top_vec_);
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = this->prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(this->prob_.count() * sizeof(Dtype),
                     prob_data, bottom_diff);
    const Dtype* tweight = tweight_.gpu_data();
    caffe_gpu_mul(tweight_.count(), bottom_diff, tweight, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const Dtype* weight = bottom[2]->gpu_data();
    const int dim = this->prob_.count() / this->outer_num_;
    const int nthreads = this->outer_num_ * this->inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = this->prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    WeightedSoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, weight,
        bottom_diff, this->outer_num_, dim, this->inner_num_,
        this->has_ignore_label_, this->ignore_label_, counts);

    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (this->normalization_ == LossParameter_NormalizationMode_VALID &&
        this->has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0]
        / this->get_normalizer(this->normalization_, valid_count);
    caffe_gpu_scal(this->prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WeightedSoftmaxWithLossLayer);

}  // namespace caffe
