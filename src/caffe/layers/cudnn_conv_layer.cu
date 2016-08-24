#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

__global__ void sync_conv_groups() {}

template<typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  size_t& WORKSPACE_SIZE = workspace_size(device);
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Sometimes closer to zero we might have memory info diverged from reality
    // If try_reserve fails, it updates the info internally and we proceed with
    // Reshape one more time
    // Note: if WORKSPACE_SIZE is already allocated next line does nothing.
    if (!WORKSPACE.try_reserve(WORKSPACE_SIZE)) {
      use_algo_seeker_ = true;
      this->Reshape(bottom, top);
      WORKSPACE.reserve(WORKSPACE_SIZE);
    }

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(Caffe::cudnn_handle(),
          cudnn::dataType<Dtype>::one,
          bottom_descs_[i], bottom_data + bottom_offset_ * g,
          filter_desc_, weight + this->weight_offset_ * g,
          conv_descs_[i],
          fwd_algo_[i], WORKSPACE.data(), WORKSPACE.size(),
          cudnn::dataType<Dtype>::zero,
          top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(Caffe::cudnn_handle(),
            cudnn::dataType<Dtype>::one,
            bias_desc_, bias_data + bias_offset_ * g,
            cudnn::dataType<Dtype>::one,
            top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamLegacy));
  }
  // Possibly use faster algorithms by allowing larger workspace.
  use_modest_workspace_ = false;
}

template<typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  size_t& WORKSPACE_SIZE = workspace_size(device);
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();

    // Sometimes closer to zero we might have memory info diverged from reality
    // If try_reserve fails, it updates the info internally and we proceed with
    // Reshape one more time.
    // Note: if WORKSPACE_SIZE is already allocated next line does nothing.
    if (!WORKSPACE.try_reserve(WORKSPACE_SIZE)) {
      use_algo_seeker_ = true;
      this->Reshape(bottom, top);
      WORKSPACE.reserve(WORKSPACE_SIZE);
    }

    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(Caffe::cudnn_handle(),
            cudnn::dataType<Dtype>::one,
            top_descs_[i], top_diff + top_offset_ * g,
            cudnn::dataType<Dtype>::one,
            bias_desc_, bias_diff + bias_offset_ * g));
      }
      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(Caffe::cudnn_handle(),
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            top_descs_[i], top_diff + top_offset_ * g,
            conv_descs_[i],
            bwd_filter_algo_[i], WORKSPACE.data(), WORKSPACE.size(),
            cudnn::dataType<Dtype>::one,
            filter_desc_, weight_diff + this->weight_offset_ * g));
      }
      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(Caffe::cudnn_handle(),
            cudnn::dataType<Dtype>::one,
            filter_desc_, weight + this->weight_offset_ * g,
            top_descs_[i], top_diff + top_offset_ * g,
            conv_descs_[i],
            bwd_data_algo_[i], WORKSPACE.data(), WORKSPACE.size(),
            cudnn::dataType<Dtype>::zero,
            bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamLegacy));
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
