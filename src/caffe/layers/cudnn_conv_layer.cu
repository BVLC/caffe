#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

// Those defines serve single purpose to keep sane C++ formatting
// in presence of <80 characters rule
#define cudnnConvFwd                       cudnnConvolutionForward
#define cudnnConvBwdBias                   cudnnConvolutionBackwardBias
#define cudnnConvBwdFilter                 cudnnConvolutionBackwardFilter_v3
#define cudnnConvBwdData                   cudnnConvolutionBackwardData_v3

namespace caffe {

  __global__ void sync_conv_groups() { }

  template <typename Dtype>
  void CuDNNConvolutionLayer<Dtype>::
  Forward_gpu(const vector<Blob<Dtype>*>& bottom,
              const vector<Blob<Dtype>*>& top) {
    const Dtype* weight = this->blobs_[0]->gpu_data();
    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* top_data = top[i]->mutable_gpu_data();


      // Forward through cuDNN in parallel over groups.
      for (int g = 0; g < this->group_; g++) {
        gpu_memory::allocate(&workspaceData, workspace_fwd_sizes_[i]);
        // Filters.
        CUDNN_CHECK(cudnnConvFwd(Caffe::cudnn_handle(),
                                 cudnn::dataType<Dtype>::one,
                                 bottom_descs_[i],
                                 bottom_data + bottom_offset_ * g,
                                 filter_desc_,
                                 weight + this->weight_offset_ * g,
                                 conv_descs_[i],
                                 fwd_algo_[i], workspaceData,
                                 workspace_fwd_sizes_[i],
                                 cudnn::dataType<Dtype>::zero,
                                 top_descs_[i],
                                 top_data + top_offset_ * g));

        gpu_memory::deallocate(workspaceData);
        workspaceData = NULL;
        // Bias.
        if (this->bias_term_) {
          const Dtype* bias_data = this->blobs_[1]->gpu_data();
          CUDNN_CHECK(cudnnAddTensor_v3(Caffe::cudnn_handle(),
                                        cudnn::dataType<Dtype>::one,
                                        bias_desc_,
                                        bias_data + bias_offset_ * g,
                                        cudnn::dataType<Dtype>::one,
                                        top_descs_[i],
                                        top_data + top_offset_ * g));
        }
      }

      // Synchronize the work across groups, each of which went into its own
      // stream, by launching an empty kernel into the default (null) stream.
      // NOLINT_NEXT_LINE(whitespace/operators)
      CUDA_CHECK(cudaStreamSynchronize(cudaStreamLegacy));
    }
  }

  template <typename Dtype>
  void
  CuDNNConvolutionLayer<Dtype>::
  Backward_gpu(const vector<Blob<Dtype>*>& top,
               const vector<bool>& propagate_down,
               const vector<Blob<Dtype>*>& bottom) {
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

    for (int i = 0; i < top.size(); ++i) {
      const Dtype* top_diff = top[i]->gpu_diff();

      // Backward through cuDNN in parallel over groups and gradients.
      for (int g = 0; g < this->group_; g++) {
        // Gradient w.r.t. bias.
        if (this->bias_term_ && this->param_propagate_down_[1]) {
          CUDNN_CHECK(cudnnConvBwdBias(Caffe::cudnn_handle(),
                                       cudnn::dataType<Dtype>::one,
                                       top_descs_[i],
                                       top_diff + top_offset_ * g,
                                       cudnn::dataType<Dtype>::one,
                                       bias_desc_,
                                       bias_diff + bias_offset_ * g));
        }

        // Gradient w.r.t. weights.
        if (this->param_propagate_down_[0]) {
          gpu_memory::allocate(&workspaceData,
                               workspace_bwd_filter_sizes_[i]);
          const Dtype* bottom_data = bottom[i]->gpu_data();
          CUDNN_CHECK(cudnnConvBwdFilter(Caffe::cudnn_handle(),
                                         cudnn::dataType<Dtype>::one,
                                         bottom_descs_[i],
                                         bottom_data + bottom_offset_ * g,
                                         top_descs_[i],
                                         top_diff + top_offset_ * g,
                                         conv_descs_[i],
                                         bwd_filter_algo_[i],
                                         workspaceData,
                                         workspace_bwd_filter_sizes_[i],
                                         cudnn::dataType<Dtype>::one,
                                         filter_desc_,
                                         weight_diff + weight_offset_ * g));
          gpu_memory::deallocate(workspaceData);
          workspaceData = NULL;
        }

        // Gradient w.r.t. bottom data.
        if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->gpu_data();
          }
          Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
          gpu_memory::allocate(&workspaceData,
                               workspace_bwd_data_sizes_[i]);
          CUDNN_CHECK(cudnnConvBwdData(Caffe::cudnn_handle(),
                                       cudnn::dataType<Dtype>::one,
                                       filter_desc_,
                                       weight + this->weight_offset_ * g,
                                       top_descs_[i],
                                       top_diff + top_offset_ * g,
                                       conv_descs_[i],
                                       bwd_data_algo_[i], workspaceData,
                                       workspace_bwd_data_sizes_[i],
                                       cudnn::dataType<Dtype>::zero,
                                       bottom_descs_[i],
                                       bottom_diff + bottom_offset_ * g));
          gpu_memory::deallocate(workspaceData);
          workspaceData = NULL;
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
