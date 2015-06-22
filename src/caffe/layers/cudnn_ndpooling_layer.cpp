#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CudnnNdPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  CHECK(pool_param.has_kernel_shape())
	<< "Kernel shape is required.";
  if(pool_param.has_pad_shape()) {
    CHECK_EQ(pool_param.kernel_shape().dim_size(), pool_param.pad_shape().dim_size())
	  << "Kernel and Pad shape don't match !";
  }
  if(pool_param.has_stride_shape()) {
    CHECK_EQ(pool_param.kernel_shape().dim_size(), pool_param.stride_shape().dim_size())
	  << "Kernel and Stride shape don't match !";
  }
  global_pooling_ = pool_param.global_pooling();

  if(global_pooling_) {
	kernel_shape_ = vector<int>(bottom[0]->shape().begin()+2, bottom[0]->shape().end());
  } else {
  	for(int i = 0; i < pool_param.kernel_shape().dim_size(); ++i) {
  	  kernel_shape_.push_back(pool_param.kernel_shape().dim(i));
  	  CHECK_GT(kernel_shape_[i], 0) << "Filter dimensions cannot be zero.";
  	}
  }
  if(pool_param.has_pad_shape()) {
    for(int i = 0; i < pool_param.kernel_shape().dim_size(); ++i) {
      pad_shape_.push_back(pool_param.pad_shape().dim(i));
    }
  } else {
    pad_shape_ = std::vector<int>(kernel_shape_.size(), 0);
  }
  if(pool_param.has_stride_shape()) {
    for(int i = 0; i < pool_param.kernel_shape().dim_size(); ++i) {
      stride_shape_.push_back(pool_param.stride_shape().dim(i));
    }
  } else {
    stride_shape_ = std::vector<int>(kernel_shape_.size(), 1);
  }

  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensorDesc<Dtype>(&bottom_desc_);
  cudnn::createTensorDesc<Dtype>(&top_desc_);
  cudnn::createNdPoolingDesc<Dtype>(&pooling_desc_,
      this->layer_param_.pooling_param().pool(), &mode_,
      kernel_shape_, pad_shape_, stride_shape_);
  handles_setup_ = true;
}

template <typename Dtype>
void CudnnNdPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  channels_ = bottom[0]->shape(1);
  input_shape_ = bottom[0]->shape();
  if(global_pooling_) {
	kernel_shape_ = vector<int>(bottom[0]->shape().begin()+2, bottom[0]->shape().end());
  }

  compute_output_shape();
  top[0]->Reshape(pooled_shape_);
 
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(pooled_shape_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(pooled_shape_);
  }

  cudnn::setTensorNdDesc<Dtype>(&bottom_desc_, input_shape_);
  cudnn::setTensorNdDesc<Dtype>(&top_desc_, pooled_shape_);
}

template <typename Dtype>
void CudnnNdPoolingLayer<Dtype>::compute_output_shape() {
  pooled_shape_ = std::vector<int>(input_shape_.begin(), input_shape_.begin()+2);
  for(int i = 2; i < input_shape_.size(); ++i) {
	int dim = (input_shape_[i] + 2 * pad_shape_[i-2] - kernel_shape_[i-2]) / stride_shape_[i-2] + 1;

	if(pad_shape_[i-2] > 0) {
      if ((dim - 1) * stride_shape_[i-2] >= input_shape_[i] + pad_shape_[i-2]) {
        --dim;
      }
      CHECK_LT((dim - 1) * stride_shape_[i-2], input_shape_[i] + pad_shape_[i-2]);
	}

	if(dim > 1) {
	  pooled_shape_.push_back(dim);
	}

  }
}

template <typename Dtype>
void CudnnNdPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom, const vector<Blob<Dtype>* >& top) {
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void CudnnNdPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>* >& bottom, const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& top) {
	NOT_IMPLEMENTED;
}

template <typename Dtype>
CudnnNdPoolingLayer<Dtype>::~CudnnNdPoolingLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyPoolingDescriptor(pooling_desc_);
  cudnnDestroy(handle_);
}

INSTANTIATE_CLASS(CudnnNdPoolingLayer);

}   // namespace caffe
#endif
