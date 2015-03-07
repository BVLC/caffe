#ifndef CONV2_HPP
#define CONV2_HPP
void conv2_gpu(const float *img,int height,int width,const float* kernel,int kernel_height,int kernel_width,
			   int pad_h,int pad_w,int stride_h,int stride_w,float *r);
void conv2_gpu(const double *img,int height,int width,const double* kernel,int kernel_height,int kernel_width,
			   int pad_h,int pad_w,int stride_h,int stride_w,double *r);
#endif