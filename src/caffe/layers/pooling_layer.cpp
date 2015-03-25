#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#if defined(USE_OPENCL)
#include <caffe/util/OpenCL/OpenCLDevice.hpp>
#include <caffe/util/OpenCL/pooling_layer.hpp>
#include <caffe/util/benchmark.hpp>
#endif

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

#if defined(USE_OPENCL)

namespace OpenCL {

template<typename T>
bool clMaxPoolBackward(
		const int nthreads,
		const T* top_diff,
		const int* mask,
		const T* top_mask,
		const int num,
		const int channels,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		const int kernel_h,
		const int kernel_w,
		const int stride_h,
		const int stride_w,
		const int pad_h,
		const int pad_w,
		T* bottom_diff) {

	std::string kernel_name = clGetKernelName<T>("MaxPoolBackward");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, nthreads)
	CL_SET_ARRAY_KERNEL_ARG(&top_diff)
	CL_SET_ARRAY_KERNEL_ARG(&mask)
	CL_SET_ARRAY_KERNEL_ARG(&top_mask)
	CL_SET_TYPE_KERNEL_ARG(int, num)
	CL_SET_TYPE_KERNEL_ARG(int, channels)
	CL_SET_TYPE_KERNEL_ARG(int, height)
	CL_SET_TYPE_KERNEL_ARG(int, width)
	CL_SET_TYPE_KERNEL_ARG(int, pooled_height)
	CL_SET_TYPE_KERNEL_ARG(int, pooled_width)
	CL_SET_TYPE_KERNEL_ARG(int, kernel_h)
	CL_SET_TYPE_KERNEL_ARG(int, kernel_w)
	CL_SET_TYPE_KERNEL_ARG(int, stride_h)
	CL_SET_TYPE_KERNEL_ARG(int, stride_w)
	CL_SET_TYPE_KERNEL_ARG(int, pad_h)
	CL_SET_TYPE_KERNEL_ARG(int, pad_w)
	CL_SET_ARRAY_KERNEL_ARG(&bottom_diff)

	size_t global = nthreads;//CAFFE_GET_GLOBAL_WORKITEMS(nthreads, OPENCL_LOCAL_SIZE);
	size_t local  = 1;//CAFFE_GET_LOCAL_WORKITEMS(nthreads, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	clFinish(*queue);
	LOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clMaxPoolBackward<float>(const int nthreads, const float* top_diff, const int* mask, const float* top_mask, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, float* bottom_diff);
template bool clMaxPoolBackward<double>(const int nthreads, const double* top_diff, const int* mask, const double* top_mask, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, double* bottom_diff);

template<typename T>
bool clAvePoolBackward(
		const int nthreads,
		const T* top_diff,
		const int num,
		const int channels,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		const int kernel_h,
		const int kernel_w,
		const int stride_h,
		const int stride_w,
		const int pad_h,
		const int pad_w,
		T* bottom_diff) {

	std::string kernel_name = clGetKernelName<T>("AvePoolBackward");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, nthreads)
	CL_SET_ARRAY_KERNEL_ARG(&top_diff)
	CL_SET_TYPE_KERNEL_ARG(int, num)
	CL_SET_TYPE_KERNEL_ARG(int, channels)
	CL_SET_TYPE_KERNEL_ARG(int, height)
	CL_SET_TYPE_KERNEL_ARG(int, width)
	CL_SET_TYPE_KERNEL_ARG(int, pooled_height)
	CL_SET_TYPE_KERNEL_ARG(int, pooled_width)
	CL_SET_TYPE_KERNEL_ARG(int, kernel_h)
	CL_SET_TYPE_KERNEL_ARG(int, kernel_w)
	CL_SET_TYPE_KERNEL_ARG(int, stride_h)
	CL_SET_TYPE_KERNEL_ARG(int, stride_w)
	CL_SET_TYPE_KERNEL_ARG(int, pad_h)
	CL_SET_TYPE_KERNEL_ARG(int, pad_w)
	CL_SET_ARRAY_KERNEL_ARG(&bottom_diff)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(nthreads, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(nthreads, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	clFinish(*queue);
	LOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clAvePoolBackward<float>(const int nthreads, const float* top_diff, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, float* bottom_diff);
template bool clAvePoolBackward<double>(const int nthreads, const double* top_diff, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, double* bottom_diff);

template<typename T>
bool clStoPoolBackward(
		const int nthreads,
		const T* rand_idx,
		const T* top_diff,
		const int num,
		const int channels,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		const int kernel_h,
		const int kernel_w,
		const int stride_h,
		const int stride_w,
		T* bottom_diff) {

	std::string kernel_name = clGetKernelName<T>("StoPoolBackward");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, nthreads)
	CL_SET_ARRAY_KERNEL_ARG(&rand_idx)
	CL_SET_ARRAY_KERNEL_ARG(&top_diff)
	CL_SET_TYPE_KERNEL_ARG(int, num)
	CL_SET_TYPE_KERNEL_ARG(int, channels)
	CL_SET_TYPE_KERNEL_ARG(int, height)
	CL_SET_TYPE_KERNEL_ARG(int, width)
	CL_SET_TYPE_KERNEL_ARG(int, pooled_height)
	CL_SET_TYPE_KERNEL_ARG(int, pooled_width)
	CL_SET_TYPE_KERNEL_ARG(int, kernel_h)
	CL_SET_TYPE_KERNEL_ARG(int, kernel_w)
	CL_SET_TYPE_KERNEL_ARG(int, stride_h)
	CL_SET_TYPE_KERNEL_ARG(int, stride_w)
	CL_SET_ARRAY_KERNEL_ARG(&bottom_diff)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(nthreads, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(nthreads, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	clFinish(*queue);
	LOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clStoPoolBackward<float>(const int nthreads, const float* rand_idx, const float* top_diff, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, float* bottom_diff);
template bool clStoPoolBackward<double>(const int nthreads, const double* rand_idx, const double* top_diff, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, double* bottom_diff);

template<typename T>
bool clMaxPoolForward(
		const int nthreads,
		const T* bottom_data,
		const int num,
		const int channels,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		const int kernel_h,
		const int kernel_w,
		const int stride_h,
		const int stride_w,
		const int pad_h,
		const int pad_w,
		T* top_data,
		int* mask,
		T* top_mask
		) {

	std::string	kernel_name = clGetKernelName<T>("MaxPoolForward");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, nthreads)
	CL_SET_ARRAY_KERNEL_ARG(&bottom_data)
	CL_SET_TYPE_KERNEL_ARG(int, num)
	CL_SET_TYPE_KERNEL_ARG(int, channels)
	CL_SET_TYPE_KERNEL_ARG(int, height)
	CL_SET_TYPE_KERNEL_ARG(int, width)
	CL_SET_TYPE_KERNEL_ARG(int, pooled_height)
	CL_SET_TYPE_KERNEL_ARG(int, pooled_width)
	CL_SET_TYPE_KERNEL_ARG(int, kernel_h)
	CL_SET_TYPE_KERNEL_ARG(int, kernel_w)
	CL_SET_TYPE_KERNEL_ARG(int, stride_h)
	CL_SET_TYPE_KERNEL_ARG(int, stride_w)
	CL_SET_TYPE_KERNEL_ARG(int, pad_h)
	CL_SET_TYPE_KERNEL_ARG(int, pad_w)
	CL_SET_ARRAY_KERNEL_ARG(&top_data)
	CL_SET_ARRAY_KERNEL_ARG(&mask)
	CL_SET_ARRAY_KERNEL_ARG(&top_mask)


	int dim = 1;
	size_t *global;
	size_t *local;

	size_t global3D[3] 	= {CAFFE_GET_GLOBAL_WORKITEMS(pooled_width, OPENCL_LOCAL_SIZE), (size_t) pooled_height, (size_t) num*channels};
	size_t local3D[3]  	= {OPENCL_LOCAL_SIZE, 1, 1};

	size_t global1D 	= nthreads;//CAFFE_GET_GLOBAL_WORKITEMS(nthreads, OPENCL_LOCAL_SIZE);
	size_t local1D  	= 1;//CAFFE_GET_LOCAL_WORKITEMS(nthreads, OPENCL_LOCAL_SIZE);

	switch(0) {
	case 1:
		dim = 3;
		global 	= &global3D[0];
		local	= &local3D[0];
		break;
	default:
		dim = 1;
		global 	= &global1D;
		local	= &local1D;
	}

	std::string function = __func__;
	err = clEnqueueNDRangeKernel(*queue, *kernel, dim, NULL, global, local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	clFinish(*queue);
	LOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clMaxPoolForward<float>(
		const int nthreads,
		const float* bottom_data,
		const int num,
		const int channels,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		const int kernel_h,
		const int kernel_w,
		const int stride_h,
		const int stride_w,
		const int pad_h,
		const int pad_w,
		float* top_data,
		int* mask,
		float* top_mask
		);
template bool clMaxPoolForward<double>(
		const int nthreads,
		const double* bottom_data,
		const int num,
		const int channels,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		const int kernel_h,
		const int kernel_w,
		const int stride_h,
		const int stride_w,
		const int pad_h,
		const int pad_w,
		double* top_data,
		int* mask,
		double* top_mask
		);

template<typename T>
bool clAvePoolForward(
		const int nthreads,
		const T* bottom_data,
		const int num,
		const int channels,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		const int kernel_h,
		const int kernel_w,
		const int stride_h,
		const int stride_w,
		const int pad_h,
		const int pad_w,
		T* top_data) {

	std::string kernel_name = clGetKernelName<T>("AvePoolForward");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, nthreads)
	CL_SET_ARRAY_KERNEL_ARG(&bottom_data)
	CL_SET_TYPE_KERNEL_ARG(int, num)
	CL_SET_TYPE_KERNEL_ARG(int, channels)
	CL_SET_TYPE_KERNEL_ARG(int, height)
	CL_SET_TYPE_KERNEL_ARG(int, width)
	CL_SET_TYPE_KERNEL_ARG(int, pooled_height)
	CL_SET_TYPE_KERNEL_ARG(int, pooled_width)
	CL_SET_TYPE_KERNEL_ARG(int, kernel_h)
	CL_SET_TYPE_KERNEL_ARG(int, kernel_w)
	CL_SET_TYPE_KERNEL_ARG(int, stride_h)
	CL_SET_TYPE_KERNEL_ARG(int, stride_w)
	CL_SET_TYPE_KERNEL_ARG(int, pad_h)
	CL_SET_TYPE_KERNEL_ARG(int, pad_w)
	CL_SET_ARRAY_KERNEL_ARG(&top_data)

	int dim = 1;
	size_t *global;
	size_t *local;

	size_t global3D[3] 	= {CAFFE_GET_GLOBAL_WORKITEMS(pooled_width, OPENCL_LOCAL_SIZE), (size_t) pooled_height, (size_t) num*channels};
	size_t local3D[3]  	= {OPENCL_LOCAL_SIZE, 1, 1};

	size_t global1D 	= CAFFE_GET_GLOBAL_WORKITEMS(nthreads, OPENCL_LOCAL_SIZE);
	size_t local1D  	= CAFFE_GET_LOCAL_WORKITEMS(nthreads, OPENCL_LOCAL_SIZE);

	switch(OPENCL_OPT_LEVEL) {
	case 1:
		dim = 3;
		global 	= &global3D[0];
		local	= &local3D[0];
		break;
	default:
		dim = 1;
		global 	= &global1D;
		local	= &local1D;
	}

	err = clEnqueueNDRangeKernel(*queue, *kernel, dim, NULL, global, local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	clFinish(*queue);
	LOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clAvePoolForward<float>(const int nthreads, const float* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, float* top_data);
template bool clAvePoolForward<double>(const int nthreads, const double* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, double* top_data);

template<typename T>
bool clStoPoolForwardTrain(
		const int nthreads,
		const T* bottom_data,
		const int num,
		const int channels,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		const int kernel_h,
		const int kernel_w,
		const int stride_h,
		const int stride_w,
		T* rand_idx,
		T* top_data) {

	std::string kernel_name = clGetKernelName<T>("StoPoolForwardTrain");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, nthreads)
	CL_SET_ARRAY_KERNEL_ARG(&bottom_data)
	CL_SET_TYPE_KERNEL_ARG(int, num)
	CL_SET_TYPE_KERNEL_ARG(int, channels)
	CL_SET_TYPE_KERNEL_ARG(int, height)
	CL_SET_TYPE_KERNEL_ARG(int, width)
	CL_SET_TYPE_KERNEL_ARG(int, pooled_height)
	CL_SET_TYPE_KERNEL_ARG(int, pooled_width)
	CL_SET_TYPE_KERNEL_ARG(int, kernel_h)
	CL_SET_TYPE_KERNEL_ARG(int, kernel_w)
	CL_SET_TYPE_KERNEL_ARG(int, stride_h)
	CL_SET_TYPE_KERNEL_ARG(int, stride_w)
	CL_SET_ARRAY_KERNEL_ARG(&rand_idx)
	CL_SET_ARRAY_KERNEL_ARG(&top_data)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(nthreads, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(nthreads, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	clFinish(*queue);
	LOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clStoPoolForwardTrain<float>(const int nthreads, const float* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, float* rand_idx, float* top_data);
template bool clStoPoolForwardTrain<double>(const int nthreads, const double* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, double* rand_idx, double* top_data);

template<typename T>
bool clStoPoolForwardTest(
		const int nthreads,
		const T* bottom_data,
		const int num,
		const int channels,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		const int kernel_h,
		const int kernel_w,
		const int stride_h,
		const int stride_w,
		T* top_data) {

	std::string kernel_name = clGetKernelName<T>("StoPoolForwardTest");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, nthreads)
	CL_SET_ARRAY_KERNEL_ARG(&bottom_data)
	CL_SET_TYPE_KERNEL_ARG(int, num)
	CL_SET_TYPE_KERNEL_ARG(int, channels)
	CL_SET_TYPE_KERNEL_ARG(int, height)
	CL_SET_TYPE_KERNEL_ARG(int, width)
	CL_SET_TYPE_KERNEL_ARG(int, pooled_height)
	CL_SET_TYPE_KERNEL_ARG(int, pooled_width)
	CL_SET_TYPE_KERNEL_ARG(int, kernel_h)
	CL_SET_TYPE_KERNEL_ARG(int, kernel_w)
	CL_SET_TYPE_KERNEL_ARG(int, stride_h)
	CL_SET_TYPE_KERNEL_ARG(int, stride_w)
	CL_SET_ARRAY_KERNEL_ARG(&top_data)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(nthreads, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(nthreads, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	clFinish(*queue);
	LOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clStoPoolForwardTest<float>(const int nthreads, const float* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, float* top_data);
template bool clStoPoolForwardTest<double>(const int nthreads, const double* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, double* top_data);
} // namespace OpenCL

template<typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    /*
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
        mask, top_mask);
    */
    BOOL_CHECK(
    		caffe::OpenCL::clMaxPoolForward(count, bottom_data, bottom[0]->num(), channels_,
					height_, width_, pooled_height_, pooled_width_, kernel_h_,
					kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
					mask, top_mask)
    );
    break;
  case PoolingParameter_PoolMethod_AVE:
  	/*
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
    */
  	BOOL_CHECK(
  			caffe::OpenCL::clAvePoolForward(count, bottom_data, bottom[0]->num(), channels_,
					height_, width_, pooled_height_, pooled_width_, kernel_h_,
					kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data)
  	);
  	break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      /*
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_,
          rand_idx_.mutable_gpu_data(), top_data);
      */
      BOOL_CHECK(
      		caffe::OpenCL::clStoPoolForwardTrain(
      				count, bottom_data, bottom[0]->num(), channels_,
      				height_, width_, pooled_height_, pooled_width_, kernel_h_,
      				kernel_w_, stride_h_, stride_w_,
      				rand_idx_.mutable_gpu_data(), top_data)
      );
    } else {
    	/*
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, top_data);
      */
    	BOOL_CHECK(
    			caffe::OpenCL::clStoPoolForwardTest(count, bottom_data, bottom[0]->num(), channels_,
              height_, width_, pooled_height_, pooled_width_, kernel_h_,
              kernel_w_, stride_h_, stride_w_, top_data)
    	);
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template<typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    /*
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        bottom_diff);
    */
    BOOL_CHECK(
    		caffe::OpenCL::clMaxPoolBackward<Dtype>(
    		        count, top_diff, mask, top_mask, top[0]->num(), channels_,
    		        height_, width_, pooled_height_, pooled_width_,
    		        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
    		        bottom_diff)
    );
    break;
  case PoolingParameter_PoolMethod_AVE:
  	/*
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
    */
    BOOL_CHECK(
    		caffe::OpenCL::clAvePoolBackward<Dtype>(
    		        count, top_diff, top[0]->num(), channels_,
    		        height_, width_, pooled_height_, pooled_width_, kernel_h_,
    		        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff)
    );
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
  	/*
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff,
        top[0]->num(), channels_, height_, width_, pooled_height_,
        pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        bottom_diff);
    */
    BOOL_CHECK(
    		caffe::OpenCL::clStoPoolBackward<Dtype>(
    				count, rand_idx_.gpu_data(), top_diff,
    				top[0]->num(), channels_, height_, width_, pooled_height_,
    				pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
    				bottom_diff)
    );
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  //CUDA_POST_KERNEL_CHECK;
}

#endif // USE_OPENCL

#if defined(CPU_ONLY) && ! defined(USE_OPENCL)
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);
REGISTER_LAYER_CLASS(Pooling);

}  // namespace caffe
