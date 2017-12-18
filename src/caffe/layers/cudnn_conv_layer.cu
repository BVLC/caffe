#ifdef USE_CUDNN
#include <chrono>
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/syncedmem.hpp"

namespace caffe {

__global__ void sync_conv_groups() {}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  Forward_const_gpu(bottom, top);
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_const_gpu(
    const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) const {

  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int *pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int *stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  int num = bottom[0]->count(0, this->channel_axis_);

  // Initialize algorithm arrays
  vector<cudnnConvolutionFwdAlgo_t> fwd_algo_(bottom.size(), {});
  vector<size_t> workspace_fwd_sizes_(bottom.size(), {});

  // initialize size arrays
  for (size_t i = 0; i < bottom.size(); ++i) {
    // initialize all to default algorithms
    fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
    // default algorithms don't require workspace
    workspace_fwd_sizes_[i] = 0;
  }

  // Specify workspace limit for kernels directly until we have a
  // planning strategy and a rewrite of Caffe's GPU memory mangagement

  if (!bottom_descs_ptr_.get()) {
    bottom_descs_ptr_.reset(new vector<cudnnTensorDescriptor_t>);
    for (int i = 0; i < bottom.size(); i++) {
      cudnnTensorDescriptor_t bottom_desc;
      cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
      bottom_descs_ptr_->push_back(bottom_desc);
    }
  }

  if (!top_descs_ptr_.get()) {
    top_descs_ptr_.reset(new vector<cudnnTensorDescriptor_t>);
    for (int i = 0; i < bottom.size(); i++) {
      cudnnTensorDescriptor_t top_desc;
      cudnn::createTensor4dDesc<Dtype>(&top_desc);
      top_descs_ptr_->push_back(top_desc);
    }
  }

  if (!conv_descs_ptr_.get()) {
    conv_descs_ptr_.reset(new vector<cudnnConvolutionDescriptor_t>);
    // Create tensor descriptor(s) for data and corresponding convolution(s).
    for (int i = 0; i < bottom.size(); i++) {
      cudnnConvolutionDescriptor_t conv_desc;
      cudnn::createConvolutionDesc<Dtype>(&conv_desc);
      conv_descs_ptr_->push_back(conv_desc);
    }
  }

  /*
  if (!handle_ptr_.get()) {
    handle_ptr_.reset(new vector<cudnnHandle_t>(this->group_, {}));


    for (int g = 0; g < this->group_; g++) {
      CUDNN_CHECK(cudnnCreate(&(*handle_ptr_)[g]));
      CUDNN_CHECK(cudnnSetStream((*handle_ptr_)[g], cudaStreamPerThread));
    }
  }
  */

  if (!filter_desc_ptr_.get()) {
    // Create filter descriptor.
    const int *kernel_shape_data = this->kernel_shape_.cpu_data();
    const int kernel_h = kernel_shape_data[0];
    const int kernel_w = kernel_shape_data[1];
    filter_desc_ptr_.reset(new cudnnFilterDescriptor_t{});
    cudnn::createFilterDesc<Dtype>(
        &(*filter_desc_ptr_), this->num_output_ / this->group_,
        this->channels_ / this->group_, kernel_h, kernel_w);
  }

  for (int i = 0; i < bottom.size(); i++) {
    cudnn::setTensor4dDesc<Dtype>(
        &(*bottom_descs_ptr_)[i], num, this->channels_ / this->group_, height,
        width, this->channels_ * height * width, height * width, width, 1);
    cudnn::setTensor4dDesc<Dtype>(
        &(*top_descs_ptr_)[i], num, this->num_output_ / this->group_,
        height_out, width_out,
        this->num_output_ * (*this->conv_out_spatial_dim_ptr_),
        (*this->conv_out_spatial_dim_ptr_), width_out, 1);
    cudnn::setConvolutionDesc<Dtype>(&(*conv_descs_ptr_)[i],
                                     (*bottom_descs_ptr_)[i], *filter_desc_ptr_,
                                     pad_h, pad_w, stride_h, stride_w);
    // choose forward and backward algorithms + workspace(s)
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
        //(*handle_ptr_)[0]
        Caffe::cudnn_handle(), (*bottom_descs_ptr_)[i], *filter_desc_ptr_,
        (*conv_descs_ptr_)[i], (*top_descs_ptr_)[i],
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwd_algo_[i]));

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        //(*handle_ptr_)[0]
        Caffe::cudnn_handle(), (*bottom_descs_ptr_)[i], *filter_desc_ptr_,
        (*conv_descs_ptr_)[i], (*top_descs_ptr_)[i], fwd_algo_[i],
        &(workspace_fwd_sizes_[i])));
  }

  // reduce over all workspace sizes to get a maximum to allocate / reallocate
  size_t total_workspace_fwd = 0;

  for (size_t i = 0; i < bottom.size(); i++) {
    total_workspace_fwd =
        std::max(total_workspace_fwd, workspace_fwd_sizes_[i]);
  }
  // get max over all operations
  size_t max_workspace = total_workspace_fwd;

  // ensure all groups have enough workspace
  size_t total_max_workspace = max_workspace * (this->group_);

  // free the existing workspace and allocate a new (larger) one
  this->workspaceData.reset(new Blob<int>(1, 1, 1, total_max_workspace));
  if (!this->workspaceData->count()) {
    // force zero memory path
    for (int i = 0; i < bottom.size(); i++) {
      workspace_fwd_sizes_[i] = 0;
      fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
    }
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {

    if (!bias_desc_ptr_.get()) {
      bias_desc_ptr_.reset(new cudnnTensorDescriptor_t);
      cudnn::createTensor4dDesc<Dtype>(&(*bias_desc_ptr_));
    }

    cudnn::setTensor4dDesc<Dtype>(&(*bias_desc_ptr_), 1,
                                  this->num_output_ / this->group_, 1, 1);
  }

  const Dtype *weight = this->blobs_[0]->gpu_data();
  int weight_offset = this->num_output_ * this->kernel_dim_ / this->group_;
  int bottom_dim = bottom[0]->count(this->channel_axis_);
  int bottom_offset = bottom_dim / this->group_;
  int top_dim = top[0]->count(this->channel_axis_);
  int top_offset = top_dim / this->group_;

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype *bottom_data = bottom[i]->gpu_data();
    Dtype *top_data = top[i]->mutable_gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      char *workspace = nullptr;

      if (workspaceData->count()) {
        workspace =
            (char *)workspaceData->mutable_gpu_data() + g * max_workspace;
      }

      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(
          //(*handle_ptr_)[g]
          Caffe::cudnn_handle(), cudnn::dataType<Dtype>::one,
          (*bottom_descs_ptr_)[i], bottom_data + bottom_offset * g,
          *filter_desc_ptr_, weight + weight_offset * g, (*conv_descs_ptr_)[i],
          fwd_algo_[i], workspace, workspace_fwd_sizes_[i],
          cudnn::dataType<Dtype>::zero, (*top_descs_ptr_)[i],
          top_data + top_offset * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype *bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(
            //(*handle_ptr_)[g],
            Caffe::cudnn_handle(), cudnn::dataType<Dtype>::one,
            (*bias_desc_ptr_), bias_data + bias_offset_ * g,
            cudnn::dataType<Dtype>::one, (*top_descs_ptr_)[i],
            top_data + top_offset * g));
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_CONST(CuDNNConvolutionLayer);

} // namespace caffe
#endif
