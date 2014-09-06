#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_STREAMS_PER_GROUP 3

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  // Initialize CUDA streams and cuNN.
  stream_         = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  handle_         = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    CUDA_CHECK(cudaStreamCreate(&stream_[g]));
    CUDNN_CHECK(cudnnCreate(&handle_[g]));
    CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
  }

  // Set the indexing parameters.
  bottom_offset_ = (this->channels_ / this->group_)
      * this->height_ * this->width_;
  top_offset_ = (this->num_output_ / this->group_)
      * this->height_out_ * this->width_out_;
  weight_offset_ = (this->num_output_ / this->group_)
      * (this->channels_ / this->group_) * this->kernel_h_ * this->kernel_w_;
  bias_offset_ = (this->num_output_ / this->group_);

  // Create filter descriptor.
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      this->kernel_h_, this->kernel_w_);

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensor4dDescriptor_t bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc,
        this->num_,
        this->channels_ / this->group_,
        this->height_, this->width_,
        this->channels_ * this->height_ * this->width_,
        this->height_ * this->width_,
        this->width_, 1);
    bottom_descs_.push_back(bottom_desc);
    cudnnTensor4dDescriptor_t top_desc;
    cudnn::createTensor4dDesc<Dtype>(&top_desc,
        this->num_,
        this->num_output_ / this->group_,
        this->height_out_, this->width_out_,
        this->num_output_ * this->height_out_ * this->width_out_,
        this->height_out_ * this->width_out_,
        this->width_out_, 1);
    top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc, bottom_desc,
        filter_desc_, this->pad_h_, this->pad_w_,
        this->stride_h_, this->stride_w_);
    conv_descs_.push_back(conv_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
  }
}

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensor4dDescriptor(bottom_descs_[i]);
    cudnnDestroyTensor4dDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensor4dDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    cudaStreamDestroy(stream_[g]);
    cudnnDestroy(handle_[g]);
  }

  delete [] stream_;
  delete [] handle_;
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}   // namespace caffe
#endif
