#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PaddingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PaddingParameter padding_param = this->layer_param_.padding_param();
  pad_beg_ = padding_param.pad_beg();
  pad_end_ = padding_param.pad_end();
  if (pad_beg_ >= 0 && pad_end_ >= 0) {
    pad_pos_ = true;
  }
  else if (pad_beg_ <= 0 && pad_end_ <= 0) {
    pad_pos_ = false;
  }
  else {
    LOG(FATAL) << "Padding must either be both positive or negative";
  }    
}

template <typename Dtype>
void PaddingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_in_ = bottom[0]->height();
  width_in_ = bottom[0]->width();
  height_out_ = height_in_ + pad_beg_ + pad_end_;
  width_out_ = width_in_ + pad_beg_ + pad_end_;
  CHECK_GT(height_out_, 0) << "Padding makes height <= 0";
  CHECK_GT(width_out_, 0) << "Padding makes width <= 0";
  top[0]->Reshape(num_, channels_, height_out_, width_out_);
}

template <typename Dtype>
void PaddingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // top[n, c, h, w] = bottom[n, c, h-pad_beg, w-pad_beg] if in range
  if (pad_pos_) {
    caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
	for (int h = 0; h < height_in_; ++h) {
	  // copy the width part
	  caffe_copy(width_in_,
	     bottom[0]->cpu_data(n, c, h),
	     top[0]->mutable_cpu_data(n, c, h + pad_beg_, pad_beg_));
	}
      }
    }
  }
  else {
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
	for (int h = 0; h < height_out_; ++h) {
	  // copy the width part
	  caffe_copy(width_out_,
	     bottom[0]->cpu_data(n, c, h - pad_beg_, - pad_beg_),
	     top[0]->mutable_cpu_data(n, c, h));
	}
      }
    }
  }
}

template <typename Dtype>
void PaddingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
  if (pad_pos_) {
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
	for (int h = 0; h < height_in_; ++h) {
	  // copy the width part
	  caffe_axpy(width_in_, (Dtype)1.,
	     top[0]->cpu_diff(n, c, h + pad_beg_, pad_beg_),
	     bottom[0]->mutable_cpu_diff(n, c, h));
	}
      }
    }
  }
  else {
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
	for (int h = 0; h < height_out_; ++h) {
	  // copy the width part
	  caffe_axpy(width_out_, (Dtype)1.,
	     top[0]->cpu_diff(n, c, h),
	     bottom[0]->mutable_cpu_diff(n, c, h - pad_beg_, - pad_beg_));
	}
      }
    }
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void PaddingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // top[n, c, h, w] = bottom[n, c, h-pad_beg, w-pad_beg] if in range
  if (pad_pos_) {
    caffe_gpu_set(top[0]->count(), Dtype(0), top[0]->mutable_gpu_data());
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
	CUDA_CHECK(cudaMemcpy2D(
	    top[0]->mutable_gpu_data(n, c, pad_beg_, pad_beg_), sizeof(Dtype) * width_out_,
	    bottom[0]->gpu_data(n, c), sizeof(Dtype) * width_in_,
	    sizeof(Dtype) * width_in_, height_in_,
	    cudaMemcpyDeviceToDevice));
      }
    }
  }
  else {
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
	CUDA_CHECK(cudaMemcpy2D(
	    top[0]->mutable_gpu_data(n, c), sizeof(Dtype) * width_out_,
	    bottom[0]->gpu_data(n, c, - pad_beg_, - pad_beg_), sizeof(Dtype) * width_in_,
	    sizeof(Dtype) * width_out_, height_out_,
	    cudaMemcpyDeviceToDevice));
      }
    }
  }
}

template <typename Dtype>
void PaddingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
  if (pad_pos_) {
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
	for (int h = 0; h < height_in_; ++h) {
	  // copy the width part
	  caffe_gpu_axpy(width_in_, (Dtype)1.,
	     top[0]->gpu_diff(n, c, h + pad_beg_, pad_beg_),
	     bottom[0]->mutable_gpu_diff(n, c, h));
	}
      }
    }
  }
  else {
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
	for (int h = 0; h < height_out_; ++h) {
	  // copy the width part
	  caffe_gpu_axpy(width_out_, (Dtype)1.,
	     top[0]->gpu_diff(n, c, h),
	     bottom[0]->mutable_gpu_diff(n, c, h - pad_beg_, - pad_beg_));
	}
      }
    }
  }
}
#endif
#ifdef CPU_ONLY
STUB_GPU(PaddingLayer);
#endif

INSTANTIATE_CLASS(PaddingLayer);
REGISTER_LAYER_CLASS(PADDING, PaddingLayer);

}  // namespace caffe
