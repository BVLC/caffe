#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

__global__ void sync_conv_groups() { }

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      Dtype alpha = 1.0;
      Dtype beta = 0.0;

      cudnnConvolutionFwdAlgo_t algo;

      // get the desired convolution algorithm
      CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[g],
        bottom_descs_[i],
        filter_desc_,
        conv_descs_[i],
        top_descs_[i],
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,  // memoryLimitInBytes,
        &algo));

      // get minimum size of the workspace needed for the desired algorithm
      size_t workspaceSizeInBytes_temp = 0;

      CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[g],
        bottom_descs_[i],
        filter_desc_,
        conv_descs_[i],
        top_descs_[i],
        algo,
        &workspaceSizeInBytes));

      if (workspaceSizeInBytes_temp > workspaceSizeInBytes) {
        workspaceSizeInBytes = workspaceSizeInBytes_temp;
        // free the existing workspace and allocate a new (larger) one
        cudaFree(this->workspace);
        cudaMalloc(&(this->workspace), workspaceSizeInBytes);
      }

      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            reinterpret_cast<void *>(&alpha),
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + weight_offset_ * g,
            conv_descs_[i],
            algo, workspace, workspaceSizeInBytes,
            reinterpret_cast<void *>(&beta),
            top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        Dtype alpha = 1.0;
        Dtype beta = 1.0;
        CUDNN_CHECK(cudnnAddTensor(handle_[g], CUDNN_ADD_SAME_C,
              reinterpret_cast<void *>(&alpha),
              bias_desc_, bias_data + bias_offset_ * g,
              reinterpret_cast<void *>(&beta),
              top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        Dtype alpha = 1.0;
        Dtype beta = 1.0;
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              reinterpret_cast<void *>(&alpha),
              top_descs_[i],  top_diff + top_offset_ * g,
              reinterpret_cast<void *>(&beta),
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype alpha = 1.0;
        Dtype beta = 1.0;
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle_[1*this->group_ + g],
              reinterpret_cast<void *>(&alpha),
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i], reinterpret_cast<void *>(&beta),
              filter_desc_, weight_diff + weight_offset_ * g));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        Dtype alpha = 1.0;
        Dtype beta = 0.0;
        CUDNN_CHECK(cudnnConvolutionBackwardData(handle_[2*this->group_ + g],
              reinterpret_cast<void *>(&alpha),
              filter_desc_, weight + weight_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i], reinterpret_cast<void *>(&beta),
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
