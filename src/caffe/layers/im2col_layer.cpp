#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void Im2colLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
      && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
      && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  CHECK((!conv_param.has_hole() && conv_param.has_hole_h()
      && conv_param.has_hole_w())
      || (!conv_param.has_hole_h() && !conv_param.has_hole_w()))
      << "hole is hole OR hole_h and hole_w are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_hole_h()) {
    hole_h_ = hole_w_ = conv_param.hole();
  } else {
    hole_h_ = conv_param.hole_h();
    hole_w_ = conv_param.hole_w();
  }
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
}

template <typename Dtype>
void Im2colLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  const int kernel_h_eff = kernel_h_ + (kernel_h_ - 1) * (hole_h_ - 1);
  const int kernel_w_eff = kernel_w_ + (kernel_w_ - 1) * (hole_w_ - 1);
  top[0]->Reshape(
      bottom[0]->num(), channels_ * kernel_h_ * kernel_w_,
      (height_ + 2 * pad_h_ - kernel_h_eff) / stride_h_ + 1,
      (width_ + 2 * pad_w_ - kernel_w_eff) / stride_w_ + 1
  );
}

template <typename Dtype>
void Im2colLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  im2col_cpu(bottom[0]->cpu_data(),
	bottom[0]->num(), channels_, height_, width_,
	kernel_h_, kernel_w_, pad_h_, pad_w_,
	stride_h_, stride_w_, hole_h_, hole_w_,
	top[0]->mutable_cpu_data());
}

template <typename Dtype>
void Im2colLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  col2im_cpu(top[0]->cpu_diff(),
	top[0]->num(), channels_, height_, width_,
        kernel_h_, kernel_w_, pad_h_, pad_w_,
	stride_h_, stride_w_, hole_h_, hole_w_,
	bottom[0]->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(Im2colLayer);
#endif

INSTANTIATE_CLASS(Im2colLayer);
REGISTER_LAYER_CLASS(Im2col);

}  // namespace caffe
