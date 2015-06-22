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
void CudnnNdConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  // Configure the kernel size, padding, stride, and inputs.
  CHECK(conv_param.has_kernel_shape())
	<< "Kernel shape is required.";
  if(conv_param.has_pad_shape()) {
    CHECK_EQ(conv_param.kernel_shape().dim_size(), conv_param.pad_shape().dim_size())
	  << "Kernel and Pad shape don't match !";
  }
  if(conv_param.has_stride_shape()) {
    CHECK_EQ(conv_param.kernel_shape().dim_size(), conv_param.stride_shape().dim_size())
	  << "Kernel and Stride shape don't match !";
  }
  for(int i = 0; i < conv_param.kernel_shape().dim_size(); ++i) {
  	kernel_shape_.push_back(conv_param.kernel_shape().dim(i));
    CHECK_GT(kernel_shape_[i], 0) << "Filter dimensions cannot be zero.";
  }
  if(conv_param.has_pad_shape()) {
    for(int i = 0; i < conv_param.kernel_shape().dim_size(); ++i) {
  	  pad_shape_.push_back(conv_param.pad_shape().dim(i));
	}
  } else {
	pad_shape_ = std::vector<int>(kernel_shape_.size(), 0);
  }
  if(conv_param.has_stride_shape()) {
    for(int i = 0; i < conv_param.kernel_shape().dim_size(); ++i) {
  	  stride_shape_.push_back(conv_param.stride_shape().dim(i));
	}
  } else {
	stride_shape_ = std::vector<int>(kernel_shape_.size(), 1);
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(1);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";

  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.convolution_param().bias_term();

  vector<int> weight_shape(kernel_shape_);
  weight_shape.insert(weight_shape.begin(), channels_ / group_);
  weight_shape.insert(weight_shape.begin(), num_output_);

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      vector<int> bias_shape(1, num_output_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // Initialize CUDA streams and cuDNN.
  stream_         = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  handle_         = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  workspaceSizeInBytes = 0;
  workspace = NULL;

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    CUDA_CHECK(cudaStreamCreate(&stream_[g]));
    CUDNN_CHECK(cudnnCreate(&handle_[g]));
    CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
  }

  // Set the indexing parameters.
  weight_shape[0] /= group_;
  weight_offset_ = 1;
  for(int i = 0; i < weight_shape.size(); ++i) {
	weight_offset_ *= weight_shape[i];
  }
  bias_offset_ = weight_shape[0];

  // Create filter descriptor.
  cudnn::createNdFilterDesc<Dtype>(&filter_desc_, weight_shape);

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensorDescriptor_t bottom_desc;
    cudnn::createTensorDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    cudnnTensorDescriptor_t top_desc;
    cudnn::createTensorDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensorDesc<Dtype>(&bias_desc_);
  }

  handles_setup_ = true;
}

template <typename Dtype>
void CudnnNdConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->shape(0);
  CHECK_EQ(bottom[0]->shape(1), channels_) << "Input size incompatible with convolution kernel.";
  input_shape_ = bottom[0]->shape();
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->shape(0))
	  << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->shape(1))
        << "Inputs must have same channels.";
  	for(int i = 0; i < bottom[0]->num_axes(); ++i) {
      CHECK_EQ(input_shape_[i], bottom[bottom_id]->shape(i)) << "Inputs must have same shape.";
  	}
  }
  // Shape the tops.
  compute_output_shape();
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(output_shape_);
  }

  conv_out_spatial_dim_ = 1;
  for(int i = 2; i < output_shape_.size(); ++i) {
  	conv_out_spatial_dim_ *= output_shape_[i];
  }

  kernel_dim_ = channels_;
  for(int i = 0; i < kernel_shape_.size(); ++i) {
  	kernel_dim_ *= kernel_shape_[i];
  }
  weight_offset_ = num_output_ * kernel_dim_ / group_ / group_;
  output_offset_ = num_output_ * conv_out_spatial_dim_ / group_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, conv_out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }

  bottom_offset_ = 1;
  for(int i = 1; i < input_shape_.size(); ++i) {
    bottom_offset_ *= input_shape_[i];
  }
  bottom_offset_ /= group_;
  top_offset_ = 1;
  for(int i = 1; i < output_shape_.size(); ++i) {
    top_offset_ *= output_shape_[i];
  }
  top_offset_ /= group_;

  vector<int> bottom_tensor_shape(input_shape_);
  bottom_tensor_shape[1] /= group_;
  vector<int> bottom_tensor_stride(input_shape_.size(), 1);
  for(int i = input_shape_.size()-2; i >= 0; --i) {
	bottom_tensor_stride[i] = input_shape_[i+1] * bottom_tensor_stride[i+1];
  }
  vector<int> top_tensor_shape(output_shape_);
  top_tensor_shape[1] /= group_;
  vector<int> top_tensor_stride(output_shape_.size(), 1);
  for(int i = output_shape_.size()-2; i >= 0; --i) {
	top_tensor_stride[i] = output_shape_[i+1] * top_tensor_stride[i+1];
  }

  for (int i = 0; i < bottom.size(); i++) {
    cudnn::setTensorNdDesc<Dtype>(&bottom_descs_[i],
        bottom_tensor_shape, bottom_tensor_stride);
    cudnn::setTensorNdDesc<Dtype>(&top_descs_[i],
        top_tensor_shape, top_tensor_stride);
    cudnn::setNdConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
        filter_desc_, pad_shape_, stride_shape_);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
  }
}

template <typename Dtype>
void CudnnNdConvolutionLayer<Dtype>::compute_output_shape() {
  output_shape_.clear();
  output_shape_.push_back(num_);
  output_shape_.push_back(num_output_);
  
  for(int i = 2; i < input_shape_.size(); ++i) {
    int dim = (input_shape_[i] + 2*pad_shape_[i-2] - kernel_shape_[i-2]) / stride_shape_[i-2] + 1;
	if(dim > 1){
	  output_shape_.push_back(dim);
	}
  }
}

template <typename Dtype>
void CudnnNdConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom, const vector<Blob<Dtype>* >& top) {
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void CudnnNdConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>* >& bottom, const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& top) {
	NOT_IMPLEMENTED;
}

template <typename Dtype>
CudnnNdConvolutionLayer<Dtype>::~CudnnNdConvolutionLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensorDescriptor(bottom_descs_[i]);
    cudnnDestroyTensorDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensorDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    cudaStreamDestroy(stream_[g]);
    cudnnDestroy(handle_[g]);
  }

  delete [] stream_;
  delete [] handle_;
}

INSTANTIATE_CLASS(CudnnNdConvolutionLayer);

}   // namespace caffe
#endif
