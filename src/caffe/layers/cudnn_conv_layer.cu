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
    const int* kernel_shape_data = this->kernel_shape_.cpu_data();
    const int kernel_h = kernel_shape_data[0];
    const int kernel_w = kernel_shape_data[1];
    const size_t workspace_limit_bytes =
      kernel_h * kernel_w * this->channels_ * sizeof(int) + 1;

    cudnnHandle_t handle = Caffe::cudnn_handle();

  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    size_t workspaceSizeInBytes_temp;

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      cudnnConvolutionFwdAlgo_t algo;

      if (!MemoryHandler::usingPool()) {
	handle = handle_[g];

      // pick the convolution algorithm
      // TODO(shelhamer) this should be done during reshape
      // TODO(shelhamer) the choice of automatic or manual algorithm picking
      // should be exposed in proto

      CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle,
        bottom_descs_[i],
        filter_desc_,
        conv_descs_[i],
        top_descs_[i],
        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes,  // memoryLimitInBytes,
        &algo));

      CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
        bottom_descs_[i],
        filter_desc_,
        conv_descs_[i],
        top_descs_[i],
        algo,
        &workspaceSizeInBytes_temp));

      }     
      else {
	workspaceSizeInBytes_temp = workspace_fwd_sizes_[i];
	algo=fwd_algo_[i];
      }							      


      if (workspaceSizeInBytes_temp > workspaceSizeInBytes) {
        workspaceSizeInBytes = workspaceSizeInBytes_temp;
        // free the existing workspace and allocate a new (larger) one
        MemoryHandler::freeGPU(&this->workspaceData);
        MemoryHandler::mallocGPU(&workspaceData, workspaceSizeInBytes);
        if (!workspaceData) {
          // force zero memory path
          algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
          workspaceSizeInBytes = 0;
        }
      }
 

     // Filters.

      CUDNN_CHECK(cudnnConvolutionForward(handle,
					  cudnn::dataType<Dtype>::one,
					  bottom_descs_[i], bottom_data + bottom_offset_ * g,
					  filter_desc_, weight + this->weight_offset_ * g,
					  conv_descs_[i],
					  algo, workspace, workspaceSizeInBytes,
					  cudnn::dataType<Dtype>::zero,
					  top_descs_[i], top_data + top_offset_ * g));
      
      if (MemoryHandler::usingPool()) {
	MemoryHandler::freeGPU(workspace[0]);	
	workspace[0] = NULL;
      }
      
      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor_v3(
              handle, 
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    if (MemoryHandler::usingPool())
       sync_conv_groups<<<1, 1>>>();		
    else
       CUDA_CHECK(cudaStreamSynchronize(cudaStreamLegacy));
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  cudnnHandle_t handle = Caffe::cudnn_handle();


  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    if (MemoryHandler::usingPool())
        caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  Dtype* bias_diff = NULL;

  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    if (!MemoryHandler::usingPool())
        caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();

    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
	if (!MemoryHandler::usingPool())
	  handle = handle_[2*this->group_ + g];
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(
						 handle,
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {

	if (MemoryHandler::usingPool())
	  MemoryHandler::mallocGPU(&workspace[0], workspace_bwd_filter_sizes_[i]);

        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter_v2(
						      handle,
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
//              bwd_filter_algo_[i], workspace[0], workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g));

	if (MemoryHandler::usingPool()) {
          MemoryHandler::freeGPU(workspace[0]);
	  workspace[0] = NULL;
      }
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
	if (MemoryHandler::usingPool()) {
	  MemoryHandler::mallocGPU(&workspace[0], workspace_bwd_data_sizes_[i]);
	}

        CUDNN_CHECK(cudnnConvolutionBackwardData_v2(handle,
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
//              bwd_data_algo_[i], workspace[0], workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
	if (MemoryHandler::usingPool()) {
           MemoryHandler::freeGPU(workspace[0]);
	   workspace[0] = NULL;
        }
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    if (!MemoryHandler::usingPool())
          sync_conv_groups<<<1, 1>>>(); 
     else 
     	  CUDA_CHECK(cudaStreamSynchronize(cudaStreamLegacy));
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
