#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void ConvolutionSKLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK(
      !conv_param.has_kernel_size()
      != !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(
      conv_param.has_kernel_size()
      || (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK(
      (!conv_param.has_pad() && conv_param.has_pad_h()
          && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK(
      (!conv_param.has_stride() && conv_param.has_stride_h()
          && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0)<< "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0)<< "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  CHECK_EQ(pad_h_, 0)<< "pad_h_ must be 0";
  CHECK_EQ(pad_w_, 0)<< "pad_w_ must be 0";
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
  if (!conv_param.has_kstride_h()) {
    kstride_h_ = kstride_w_ = conv_param.kstride();
  } else {
    kstride_h_ = conv_param.kstride_h();
    kstride_w_ = conv_param.kstride_w();
  }
  group_ = this->layer_param_.convolution_param().group();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num())<< "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
    << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
    << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
    << "Inputs must have same width.";
  }
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  CHECK_EQ(channels_ % group_, 0);
  // The im2col result buffer would only hold one image at a time to avoid
  // overly large memory usage.
  int ext_kernel_h = (kernel_h_ - 1) * kstride_h_ + 1;
  int ext_kernel_w = (kernel_w_ - 1) * kstride_w_ + 1;
  int height_out = (height_ - ext_kernel_h) / stride_h_ + 1;
  int width_out = (width_ - ext_kernel_w) / stride_w_ + 1;

  // TODO: Change this
  if (kstride_h_ != 23 || this->device_context_.backend() == BACKEND_CUDA) {
    col_buffer_.Reshape(1, channels_ * kernel_h_ * kernel_w_, height_out,
                        width_out);
  }
  // Set the parameters
  CHECK_EQ(num_output_ % group_, 0)
  << "Number of output should be multiples of group.";
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  // Figure out the dimensions for individual gemms.
  M_ = num_output_ / group_;
  K_ = channels_ * kernel_h_ * kernel_w_ / group_;
  N_ = height_out * width_out;
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out, width_out);
  }
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(
        new Blob<Dtype>(num_output_, channels_ / group_, kernel_h_, kernel_w_,
                        this->device_context_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(
        GetFiller<Dtype>(
            this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(
          new Blob<Dtype>(1, 1, 1, num_output_, this->device_context_));
      shared_ptr<Filler<Dtype> > bias_filler(
          GetFiller<Dtype>(
              this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Set up the all ones "bias multiplier" for adding bias using blas
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, N_);
    caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Dtype>
void ConvolutionSKLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  LayerSetUp(bottom, top);
}

template<typename Dtype>
void ConvolutionSKLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  LOG(FATAL)<< "Foward_cpu() not implemented for ConvlutionSKLayer.";
}

template<typename Dtype>
void ConvolutionSKLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL)<< " Backward_cpu() not implemented for ConvolutionSKLayer.";
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionSKLayer);
#endif

INSTANTIATE_CLASS(ConvolutionSKLayer);
REGISTER_LAYER_CLASS(ConvolutionSK);

}  // namespace caffe
