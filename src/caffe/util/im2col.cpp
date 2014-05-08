// Copyright 2013 Yangqing Jia

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <algorithm>
#include <cstdio>
#include <unistd.h>

#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#define THREAD_NUM 4

template <typename Dtype>
struct im2col_data
{
	const Dtype* data_im;
	Dtype *data_col;
	int channels,height,width,ksize,pad,stride;
	int thread_tid;
};

template <typename Dtype>
struct col2im_data
{
	Dtype* data_im;
	const Dtype *data_col;
	int channels,height,width,ksize,pad,stride;
	int thread_tid;
};

namespace caffe {

template <typename Dtype>
void* im2col_cpu_pthread(void* parameter) {
	struct im2col_data<Dtype>* param = reinterpret_cast<struct im2col_data<Dtype>*>(parameter);

    int height_col = (param->height + 2 * param->pad - param->ksize) / param->stride + 1;
    int width_col = (param->width + 2 * param->pad - param->ksize) / param->stride + 1;
    int channels_col = param->channels * param->ksize * param->ksize;
    
    for (int c = param->thread_tid; c < channels_col; c=c+THREAD_NUM) {
		int w_offset = c % param->ksize;
		int h_offset = (c / param->ksize) % param->ksize;
		int c_im = c / param->ksize / param->ksize;
		int h_pad = h_offset- param->pad;
		int w_pad = w_offset- param->pad;
	
		for (int h = 0; h < height_col; ++h) {
			for (int w = 0; w < width_col; ++w) {
			h_pad+=param->stride;
			w_pad+=param->stride;
			if (h_pad >= 0 && h_pad < param->height && w_pad >= 0 && w_pad < param->width)
			  param->data_col[(c * height_col + h) * width_col + w] =
				param->data_im[(c_im * param->height + h_pad) * param->width + w_pad];
			else
			  param->data_col[(c * height_col + h) * width_col + w] = 0;
		  }
		}
	}
}

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col) {
  
  pthread_t thread_[THREAD_NUM];
  struct im2col_data<Dtype>* param=(struct im2col_data<Dtype>*)malloc(sizeof(struct im2col_data<Dtype>));
  param->data_im=data_im;
  param->data_col=data_col;
  param->channels=channels;
  param->height=height;
  param->width=width;
  param->ksize=ksize;
  param->pad=pad;
  param->stride=stride;

  for(int i=0; i<THREAD_NUM; i++)
  {
	  param->thread_tid=i; 
	  CHECK(!pthread_create(&thread_[i], NULL,im2col_cpu_pthread<Dtype>,(void*)param)) << "Pthread execution failed.";
  }
  for(int i=0; i<THREAD_NUM; i++)
	  CHECK(!pthread_join(thread_[i], NULL)) << "Pthread joining failed.";
}

// Explicit instantiation
template void* im2col_cpu_pthread<float>(void* parameter);
template void* im2col_cpu_pthread<double>(void* parameter);
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, double* data_col);

template <typename Dtype>
void* col2im_cpu_pthread(void* parameter) {
	struct col2im_data<Dtype>* param = reinterpret_cast<struct col2im_data<Dtype>*>(parameter);

    int height_col = (param->height + 2 * param->pad - param->ksize) / param->stride + 1;
    int width_col = (param->width + 2 * param->pad - param->ksize) / param->stride + 1;
    int channels_col = param->channels * param->ksize * param->ksize;

    for (int c = param->thread_tid; c < channels_col; c=c+THREAD_NUM) {
		int w_offset = c % param->ksize;
		int h_offset = (c / param->ksize) % param->ksize;
		int c_im = c / param->ksize / param->ksize;
		int h_pad = h_offset- param->pad;
		int w_pad = w_offset- param->pad;

		for (int h = 0; h < height_col; ++h) {
			for (int w = 0; w < width_col; ++w) {
			h_pad+=param->stride;
			w_pad+=param->stride;
			if (h_pad >= 0 && h_pad < param->height && w_pad >= 0 && w_pad < param->width)
			  param->data_im[(c_im * param->height + h_pad) * param->width + w_pad] +=
			  param->data_col[(c * height_col + h) * width_col + w];
		  }
		}
	}
}

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_im) {
  memset(data_im, 0, sizeof(Dtype) * height * width * channels);
  pthread_t thread_[THREAD_NUM];
  struct col2im_data<Dtype>* param=(struct col2im_data<Dtype>*)malloc(sizeof(struct col2im_data<Dtype>));
  param->data_im=data_im;
  param->data_col=data_col;
  param->channels=channels;
  param->height=height;
  param->width=width;
  param->ksize=ksize;
  param->pad=pad;
  param->stride=stride;

  for(int i=0; i<THREAD_NUM; i++)
  {
	  param->thread_tid=i; 
	  CHECK(!pthread_create(&thread_[i], NULL,col2im_cpu_pthread<Dtype>,(void*)param)) << "Pthread execution failed.";
  }
  for(int i=0; i<THREAD_NUM; i++)
	  CHECK(!pthread_join(thread_[i], NULL)) << "Pthread joining failed.";
}

// Explicit instantiation
template void* col2im_cpu_pthread<float>(void* parameter);
template void* col2im_cpu_pthread<double>(void* parameter);
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, double* data_im);

}  // namespace caffe
