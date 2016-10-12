#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  // Initialize algorithm arrays
  fwd_algo_       = new cudnnConvolutionFwdAlgo_t[bottom.size()];
  bwd_filter_algo_= new cudnnConvolutionBwdFilterAlgo_t[bottom.size()];
  bwd_data_algo_  = new cudnnConvolutionBwdDataAlgo_t[bottom.size()];

  // initialize size arrays
  workspace_fwd_sizes_ = new size_t[bottom.size()];
  workspace_bwd_filter_sizes_ = new size_t[bottom.size()];
  workspace_bwd_data_sizes_ = new size_t[bottom.size()];

  // Initializing algorithms and workspaces
  // Do not rely on initialized algorithms (Reshape will set algorithms
  // with correct values in the first iteration).
  for (size_t i = 0; i < bottom.size(); ++i) {
    fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)1;
    bwd_filter_algo_[i] = (cudnnConvolutionBwdFilterAlgo_t)1;
    bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t)1;
    workspace_fwd_sizes_[i] = 0;
    workspace_bwd_data_sizes_[i] = 0;
    workspace_bwd_filter_sizes_[i] = 0;
  }

  // Set the indexing parameters.
  bias_offset_ = (this->num_output_ / this->group_);

  // Create filter descriptor.
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      kernel_h, kernel_w);

  this->weight_offset_ = (this->num_output_ / this->group_) *
      (this->channels_ / this->group_) * kernel_h * kernel_w;
  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensorDescriptor_t bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);

    cudnnTensorDescriptor_t top_desc;
    cudnn::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);

    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);

    cudnnTensorDescriptor_t cached_bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&cached_bottom_desc);
    cached_bottom_descs_.push_back(cached_bottom_desc);

    cudnnConvolutionDescriptor_t cached_conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&cached_conv_desc);
    cached_conv_descs_.push_back(cached_conv_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
  }

  handles_setup_ = true;
  // When true, Reshape asks cuDNN (either Get ot FindEx) for the best algorithm
  use_algo_seeker_ = false;
  // When true, a small amount of workspace is allowed for algorithms
  use_modest_workspace_ = true;
  // When true, Reshape sets descriptors, algorithms, workspaces.
  use_reshape_ = true;
  // When true, cached bottom and conv descriptors need to be set.
  initialized_cached_descs_ = false;
}

template <typename Dtype>
size_t CuDNNConvolutionLayer<Dtype>::ComputeFindExWorkspaceSize() {
  size_t workspace_limit_bytes, total_memory, workspace_bytes;
  GPUMemory::GetInfo(&workspace_limit_bytes, &total_memory);
  // Use 95% of available memory.
  // Using all of memory may result in failure of workspace.reserve.
  // TODO: Since 95% of memory might be too large, we can allocate
  //       exactly how much FindEx needs by taking the maximum
  //       workspace among all algorithms (requires an initial call
  //       to FindEx with workspace size 0).
  workspace_bytes = workspace_limit_bytes * MAX_WORKSPACE_RATIO;
  const size_t weights_size = this->weight_offset_ * sizeof(Dtype);
  if (workspace_bytes >= weights_size) {
    workspace_bytes -= weights_size;
  } else {
    return 0UL;
  }
  return workspace_bytes;
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Check whether cached descriptors have been initialized.
  if (initialized_cached_descs_) {
    // Check whether bottom and conv descriptors have changed,
    // which then requires a new reshape and set algo.
    if ((IsBottomDescChanged(bottom)) ||
        (IsConvDescChanged(bottom))) {
      use_reshape_ = true;
      // When reshape, algorithms need to be set again.
      use_algo_seeker_ = true;
      use_modest_workspace_ = true;
    } else {
      // When no reshape is needed, setting algo may be still needed
      // (for example, if we are at iteration 1).
      // If we want to set algos, we have to use reshape in
      // current implementation.
      use_reshape_ = use_algo_seeker_;
    }
  } else {
    // If cached descriptors are not initialized yet, need to
    // do reshape which also initializes cached descriptors.
    use_reshape_ = true;
  }
  if (!use_reshape_) {
    return;
  }

  ConvolutionLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(2, this->num_spatial_axes_)
      << "CuDNNConvolution input must have 2 spatial axes "
      << "(e.g., height and width). "
      << "Use 'engine: CAFFE' for general ND convolution.";

  bottom_offset_ = this->bottom_dim_ / this->group_;
  top_offset_ = this->top_dim_ / this->group_;
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  // Set cuDNN tensor and convolution descriptors
  for (int i = 0; i < bottom.size(); i++) {
    cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
        this->num_,
        this->channels_ / this->group_, height, width,
        this->channels_ * height * width,
        height * width, width, 1);
    cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
        this->num_,
        this->num_output_ / this->group_, height_out, width_out,
        this->num_output_ * this->out_spatial_dim_,
        this->out_spatial_dim_, width_out, 1);
    cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
        filter_desc_, pad_h, pad_w, stride_h, stride_w);
    // Set cached descriptors
    cudnn::setTensor4dDesc<Dtype>(&cached_bottom_descs_[i],
        this->num_,
        this->channels_ / this->group_, height, width,
        this->channels_ * height * width,
        height * width, width, 1);
    cudnn::setConvolutionDesc<Dtype>(&cached_conv_descs_[i],
        cached_bottom_descs_[i],
        filter_desc_, pad_h, pad_w, stride_h, stride_w);
  }
  initialized_cached_descs_ = true;

  // Ask cuDNN to find the best algorithm
  if (use_algo_seeker_) {
    // FindEx: A workspace of size workspace_bytes is allocated for FindEx.
    //         Besides, workspace, a buffer is allocated for the output of
    //         FindEx-backward-filter. The size of buffer is as big as weights.
    // Get: workspace_bytes is only used as a workspace limit by Get.
    //      (no allocation happens before Get or by Get).
    size_t workspace_bytes;
    if (use_modest_workspace_) {
      // In iteration 0, use a small amount of memory in order to leave
      // most of memory for allocating layer blobs.
      workspace_bytes = INITIAL_WORKSPACE_SIZE;
    } else {
      workspace_bytes = ComputeFindExWorkspaceSize();
      // Sometimes closer to zero we might have memory info diverged from
      // reality. If try_reserve fails, it updates the info internally and
      // we have to re-evaluate the workspace size.
      if (!WORKSPACE.try_reserve(workspace_bytes)) {
        workspace_bytes = ComputeFindExWorkspaceSize();
      }
      // Avoid seeking for an algorithm in subsequent iterations
      use_algo_seeker_ = false;
    }
    // FindEx was introduced in cudnn v5.
    // If cudnn is older than v5, use Get no matter what the
    // value of cudnn_convolution_algo_seeker is.
#if CUDNN_VERSION_MIN(5, 0, 0)
    switch (this->layer_param_.convolution_param().
            cudnn_convolution_algo_seeker()) {
      case ConvolutionParameter_CuDNNConvolutionAlgorithmSeeker_GET:
        this->GetConvAlgo(bottom, top, workspace_bytes);
        break;
      case ConvolutionParameter_CuDNNConvolutionAlgorithmSeeker_FINDEX:
        WORKSPACE.reserve(workspace_bytes);
        this->FindExConvAlgo(bottom, top);
        break;
      default:
        LOG(ERROR) << "Wrong value for cudnn_convolution_algo_seeker";
        return;
    }
#else
    this->GetConvAlgo(bottom, top, workspace_bytes);
#endif
  }

  // At this point, the algorithms and their workspace are set.
  // Still need to query cuDNN for workspace size to check whether the
  // selected algorithms are valid because:
  // FindEx may return success while giving no valid algorithm as there
  // may be no algorithm available for given parameters.
  for (int i = 0; i < bottom.size(); i++) {
    // forward algorithm
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(Caffe::cudnn_handle(),
        bottom_descs_[i], filter_desc_, conv_descs_[i], top_descs_[i],
        fwd_algo_[i], &(workspace_fwd_sizes_[i])));
    // backward filter algorithm
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        Caffe::cudnn_handle(),
        bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
        bwd_filter_algo_[i], &workspace_bwd_filter_sizes_[i]));
    // backward data algorithm
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        Caffe::cudnn_handle(),
        filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
        bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]));
  }
  UpdateWorkspaceDemand(bottom.size());  // update WORKSPACE_SIZE

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::GetConvAlgo(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const size_t workspace_bytes) {

  for (int i = 0; i < bottom.size(); i++) {
    // Get forward algorithm
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(Caffe::cudnn_handle(),
        bottom_descs_[i], filter_desc_, conv_descs_[i], top_descs_[i],
        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        workspace_bytes, &fwd_algo_[i]));
    // Get backward filter algorithm
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
        Caffe::cudnn_handle(),
        bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
        CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
        workspace_bytes, &bwd_filter_algo_[i]));
    // Get backward data algorithm
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
        Caffe::cudnn_handle(),
        filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
        CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        workspace_bytes, &bwd_data_algo_[i]));
  }
}

#if CUDNN_VERSION_MIN(5, 0, 0)
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::FindExConvAlgo(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // Number of algorithms we want to consider
  // Since we only consider one algorithm (the fastest), set this to 1
  const int kRequestAlgoCount = 1;
  int fwd_algo_count;
  int filter_algo_count;
  int data_algo_count;

  cudnnConvolutionFwdAlgoPerf_t       fwd_results[kRequestAlgoCount];
  cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_results[kRequestAlgoCount];
  cudnnConvolutionBwdDataAlgoPerf_t   bwd_data_results[kRequestAlgoCount];

  // Allocate temporary buffer for weights used for backward filter FindEx
  void *tmp_weights;
  const int tmp_weights_size = sizeof(Dtype) * this->weight_offset_;
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaStream_t stream = GPUMemory::device_stream(device);
  GPUMemory::allocate(&tmp_weights, tmp_weights_size, device, stream);

  for (int i = 0; i < bottom.size(); i++) {
    // Find forward algorithm
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
                  Caffe::cudnn_handle(),
                  bottom_descs_[i],
                  bottom[i]->gpu_data(),
                  filter_desc_,
                  this->blobs_[0]->gpu_data(),
                  conv_descs_[i],
                  top_descs_[i],
                  top[i]->mutable_gpu_data(),
                  kRequestAlgoCount,
                  &fwd_algo_count,
                  fwd_results,
                  WORKSPACE.data(),
                  WORKSPACE.size()));
    fwd_algo_[i] = fwd_results[0].algo;
    workspace_fwd_sizes_[i] = fwd_results[0].memory;

    // Only set backward-filter/data algorithms in training phase
    if (this->phase_ == TRAIN) {
      // Find backward filter algorithm
      CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithmEx(
                    Caffe::cudnn_handle(),
                    bottom_descs_[i],
                    bottom[i]->gpu_data(),
                    top_descs_[i],
                    top[i]->gpu_diff(),
                    conv_descs_[i],
                    filter_desc_,
                    tmp_weights,
                    kRequestAlgoCount,
                    &filter_algo_count,
                    bwd_filter_results,
                    WORKSPACE.data(),
                    WORKSPACE.size()));
      bwd_filter_algo_[i] = bwd_filter_results[0].algo;
      workspace_bwd_filter_sizes_[i] = bwd_filter_results[0].memory;

      // Find backward data algorithm
      CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithmEx(
                    Caffe::cudnn_handle(),
                    filter_desc_,
                    this->blobs_[0]->gpu_data(),
                    top_descs_[i],
                    top[i]->gpu_diff(),
                    conv_descs_[i],
                    bottom_descs_[i],
                    bottom[i]->mutable_gpu_diff(),
                    kRequestAlgoCount,
                    &data_algo_count,
                    bwd_data_results,
                    WORKSPACE.data(),
                    WORKSPACE.size()));

      bwd_data_algo_[i] = bwd_data_results[0].algo;
      workspace_bwd_data_sizes_[i] = bwd_data_results[0].memory;
    }
  }
  GPUMemory::deallocate(tmp_weights, device, stream);
}
#endif

// Checked if there is a difference between the corresponding descriptors in
// cached_bottom_descs_ and bottom_descs_.
// No need to compare all parameters: batchsize, height, and width are enough.
template <typename Dtype>
bool CuDNNConvolutionLayer<Dtype>::IsBottomDescChanged(
  const vector<Blob<Dtype>*>& bottom) {
  int cached_n; int cached_c; int cached_h; int cached_w;
  int cached_stride_n; int cached_stride_c;
  int cached_stride_h; int cached_stride_w;
  int n; int c; int h; int w;
  cudnnDataType_t type;

  for (int i = 0; i < bottom.size(); i++) {
    CUDNN_CHECK(cudnnGetTensor4dDescriptor(
      cached_bottom_descs_[i],
      &type,
      &cached_n, &cached_c, &cached_h, &cached_w,
      &cached_stride_n, &cached_stride_c,
      &cached_stride_h, &cached_stride_w));
    const vector<int>& shape = bottom[i]->shape();
    n = shape[0];
    c = shape[1] / this->group_;
    h = shape[2];
    w = shape[3];

    if ((cached_n != n) ||
        (cached_c != c) ||
        (cached_h != h) ||
        (cached_w != w)) {
      return true;
    }
  }
  return false;
}


// Checked if there is a difference between the corresponding descriptors in
// cached_conv_descs_ and conv_descs_.
// No need to compare all parameters; pads, strides, and upscales are enough.
template <typename Dtype>
bool CuDNNConvolutionLayer<Dtype>::IsConvDescChanged(
  const vector<Blob<Dtype>*>& bottom) {
  int cached_padA[2];
  int padA[2];
  int cached_strideA[2];
  int strideA[2];
  int cached_upscaleA[2];
  int upscaleA[2];
  int arrayLength;
  cudnnConvolutionMode_t mode;
  cudnnDataType_t type;

  for (int i = 0; i < bottom.size(); i++) {
    CUDNN_CHECK(cudnnGetConvolutionNdDescriptor(
      cached_conv_descs_[i],
      2,
      &arrayLength,
      cached_padA,
      cached_strideA,
      cached_upscaleA,
      &mode,
      &type));
    CUDNN_CHECK(cudnnGetConvolutionNdDescriptor(
      conv_descs_[i],
      2,
      &arrayLength,
      padA,
      strideA,
      upscaleA,
      &mode,
      &type));

    if ((cached_padA[0] != padA[0]) ||
        (cached_padA[1] != padA[1]) ||
        (cached_strideA[0]  != strideA[0])  ||
        (cached_strideA[1]  != strideA[1])  ||
        (cached_upscaleA[0] != upscaleA[0]) ||
        (cached_upscaleA[1] != upscaleA[1])) {
      return true;
    }
  }
  return false;
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::UpdateWorkspaceDemand(int size) {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  size_t& WORKSPACE_SIZE = workspace_size(device);
  // Updating the maximum WORKSPACE_SIZE
  for (int i = 0; i < size; ++i) {
    if (workspace_fwd_sizes_[i] > WORKSPACE_SIZE) {
      WORKSPACE_SIZE = workspace_fwd_sizes_[i];
    }
    if (workspace_bwd_filter_sizes_[i] > WORKSPACE_SIZE) {
      WORKSPACE_SIZE = workspace_bwd_filter_sizes_[i];
    }
    if (workspace_bwd_data_sizes_[i] > WORKSPACE_SIZE) {
      WORKSPACE_SIZE = workspace_bwd_data_sizes_[i];
    }
  }
  // We might grab too much before calling Get/FindEx.
  // Reserve the only amount needed.
  if (WORKSPACE_SIZE < WORKSPACE.size() && !use_modest_workspace_) {
    WORKSPACE.release();
    WORKSPACE.reserve(WORKSPACE_SIZE);
  }  // else: reserve in Fwd/Bwd calls
}

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
  WORKSPACE.release();
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_descs_[i]));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_descs_[i]));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_descs_[i]));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(cached_bottom_descs_[i]));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(cached_conv_descs_[i]));
  }
  if (this->bias_term_) {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc_));
  }
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));

  delete [] fwd_algo_;
  delete [] bwd_filter_algo_;
  delete [] bwd_data_algo_;
  delete [] workspace_fwd_sizes_;
  delete [] workspace_bwd_data_sizes_;
  delete [] workspace_bwd_filter_sizes_;
}

template <typename Dtype>
size_t& CuDNNConvolutionLayer<Dtype>::workspace_size(int device) {
  if (device < 0) {
    CUDA_CHECK(cudaGetDevice(&device));
  }
  if (device + 1 > WORKSPACE_SIZES.size()) {
    WORKSPACE_SIZES.resize(device + 1);
  }
  return WORKSPACE_SIZES[device];
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}   // namespace caffe
#endif
