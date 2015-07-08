#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/benchmark.hpp"

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
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }

  height_out_ = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  width_out_  = (width_  + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  //this->setupMaskIM2COL();
  //this->setupMaskCOL2IM();
}

template <typename Dtype>
void Im2colLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  DLOG(INFO) << "BTM shape = ( "<<bottom[0]->num()<<", "<<channels_<<", "<<height_<<", "<<width_<<" )";
  DLOG(INFO) << "TOP shape = ( "<<bottom[0]->num()<<", "<<channels_ * kernel_h_ * kernel_w_<<", "<<(height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1<<", "<<(width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1<<" )";

  top[0]->Reshape(
      bottom[0]->num(), channels_ * kernel_h_ * kernel_w_,
      (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1,
      (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1);

  height_out_ = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  width_out_  = (width_  + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  this->setupMaskIM2COL();
  this->setupMaskCOL2IM();
}

template <typename Dtype>
bool Im2colLayer<Dtype>::setupMaskIM2COL() {

  if ( height_*width_ <= 0 ) {
    LOG(WARNING)<<"skipping setup on height_ = "<<height_<<" width_ = "<<width_;
    return false;
  }

  index_mask_.Reshape(1, 1, height_, width_);
  im2col_mask_.Reshape(1, channels_*kernel_h_*kernel_w_, height_out_, width_out_);
  col2im_mask_.Reshape(1, 1, height_, width_);

  for( int pixel = 0; pixel < height_*width_; pixel++ ) {
    index_mask_.mutable_cpu_data()[pixel] = pixel;
  }

  //iSNAPSHOT("index mask", index_mask_.cpu_data(), height_*width_);
  DLOG(INFO)<<"call im2col_cpu()";
  im2col_cpu(index_mask_.cpu_data(), channels_, height_,
            width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
            stride_h_, stride_w_, im2col_mask_.mutable_cpu_data());

  return true;
}

template <typename Dtype>
bool Im2colLayer<Dtype>::setupMaskCOL2IM() {

  LOG(WARNING)<<"Waiting for implementation";
  return false;
}


template <typename Dtype>
void Im2colLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
        width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
        stride_h_, stride_w_, top_data + top[0]->offset(n));
  }
}

template <typename Dtype>
void Im2colLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int n = 0; n < top[0]->num(); ++n) {
    col2im_cpu(top_diff + top[0]->offset(n), channels_, height_, width_,
        kernel_h_, kernel_w_, pad_h_, pad_w_,
        stride_h_, stride_w_, bottom_diff + bottom[0]->offset(n));
  }
}

template <typename Dtype>
bool Im2colLayer<Dtype>::hasHKernelOverlap() {

  if ( getHKernelOverlap() > 0 ) {
    return true;
  }
  return false;
}

template <typename Dtype>
bool Im2colLayer<Dtype>::hasWKernelOverlap() {

  if ( getWKernelOverlap() > 0 ) {
    return true;
  }
  return false;
}

template <typename Dtype>
bool Im2colLayer<Dtype>::hasKernelOverlap() {
  return (this->hasHKernelOverlap() || this->hasWKernelOverlap());
}

template <typename Dtype>
int Im2colLayer<Dtype>::getHKernelOverlap() {

  return kernel_h_ - stride_h_;
}

template <typename Dtype>
int Im2colLayer<Dtype>::getWKernelOverlap() {

  return kernel_w_ - stride_w_;
}

#if defined(USE_OPENCL)

	template<typename Dtype>
	void Im2colLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = (top)[0]->mutable_gpu_data();

		if ( this->hasKernelOverlap() ) {
		  DLOG(INFO)<<"KernelOverlap = "<<Im2colLayer<Dtype>::getHKernelOverlap()<<" rows and "<<Im2colLayer<Dtype>::getWKernelOverlap()<<" cols";
		}

		int level = 1;
		switch (level) {
			case 1:
			{
			  // all at once with clim2col_perf3 kernel
				int bottom_step = bottom[0]->offset(1);
				int top_step = (top)[0]->offset(1);
				im2col_group_gpu(bottom_data, bottom_step, bottom[0]->num(), channels_, height_, width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, top_data, top_step);

			  // all at once with mask
        //im2col_group_gpu(bottom_data, this->im2col_mask_.gpu_data(), bottom[0]->num(), channels_, height_, width_, kernel_h_, kernel_w_, height_out_, width_out_, top_data);
			}
				break;
			default:
				for (int n = 0; n < bottom[0]->num(); ++n) {
					// image by image using clim2col kernel
				  //im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_, width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, top_data + (top)[0]->offset(n));

				  // image by image using clim2col_perf4 kernel
				  im2col_gpu(bottom_data, bottom[0]->offset(n), channels_, height_, width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, top_data, (top)[0]->offset(n));
				}
				break;
		}
	}

	template<typename Dtype>
	void Im2colLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = (bottom)[0]->mutable_gpu_diff();

		switch (OPENCL_OPT_LEVEL) {
			case 1:
			{
				int bottom_step = (bottom)[0]->offset(1);
				int top_step = top[0]->offset(1);
				col2im_gpu(top_diff, top_step, top[0]->num(), channels_, height_, width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, bottom_diff, bottom_step);
			}
				break;
			default:
				for (int n = 0; n < top[0]->num(); ++n) {
					col2im_gpu(top_diff + top[0]->offset(n), channels_, height_, width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, bottom_diff + (bottom)[0]->offset(n));
				}
				break;
		}
	}

#endif

#if defined(CPU_ONLY) && ! defined(USE_OPENCL)
STUB_GPU(Im2colLayer);
#endif

INSTANTIATE_CLASS(Im2colLayer);
REGISTER_LAYER_CLASS(Im2col);

}  // namespace caffe
