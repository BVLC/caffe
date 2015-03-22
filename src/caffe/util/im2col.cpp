#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col) {
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_im) {
  caffe_set(height * width * channels, Dtype(0), data_im);
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int channels_col = channels * patch_h * patch_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % patch_w;
    int h_offset = (c / patch_w) % patch_h;
    int c_im = c / patch_h / patch_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im);

#if defined(USE_OPENCL)

	namespace OpenCL {

		template<typename T> bool clim2col_gpu(const int n, const T* data_im, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,
				const int height_col, const int width_col, T* data_col) {

			std::string kernel_name = clGetKernelName<T>("clim2col");

			queue = gpu->getQueue();
			if (!queue) {
				LOG(ERROR)<< gpu->name() << "> failed to get OpenCL command queue";
				return false;
			}

			kernel = gpu->getKernel(kernel_name);
			if (kernel == NULL) {
				return false;
			}

			CL_SET_KERNEL_ARG
			CL_SET_TYPE_KERNEL_ARG(int, n)
			CL_SET_ARRAY_KERNEL_ARG(&data_im)
			CL_SET_TYPE_KERNEL_ARG(int, height)
			CL_SET_TYPE_KERNEL_ARG(int, width)
			CL_SET_TYPE_KERNEL_ARG(int, kernel_h)
			CL_SET_TYPE_KERNEL_ARG(int, kernel_w)
			CL_SET_TYPE_KERNEL_ARG(int, pad_h)
			CL_SET_TYPE_KERNEL_ARG(int, pad_w)
			CL_SET_TYPE_KERNEL_ARG(int, stride_h)
			CL_SET_TYPE_KERNEL_ARG(int, stride_w)
			CL_SET_TYPE_KERNEL_ARG(int, height_col)
			CL_SET_TYPE_KERNEL_ARG(int, width_col)
			CL_SET_ARRAY_KERNEL_ARG(&data_col)

			size_t global 	= CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
			size_t local 	= CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

			err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
			if (err != CL_SUCCESS) {
				LOG(ERROR)<< "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
				return false;
			}
			clFinish(*queue);
			LOG(INFO)<< "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

			CL_SET_KERNEL_ARG_END

			return true;
		}
		template bool clim2col_gpu<float>(const int n, const float* data_im, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, float* data_col);
		template bool clim2col_gpu<double>(const int n, const double* data_im, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,	const int height_col, const int width_col, double* data_col);

		template<typename T> bool clim2col_gpu(const int n, const T* data_im, const int data_im_step, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, T* data_col, const int data_col_step) {

			std::string kernel_name = clGetKernelName<T>("clim2col_perf2");

			queue = gpu->getQueue();
			if (!queue) {
				LOG(ERROR)<< gpu->name() << "> failed to get OpenCL command queue";
				return false;
			}

			kernel = gpu->getKernel(kernel_name);
			if (kernel == NULL) {
				return false;
			}

			CL_SET_KERNEL_ARG
			CL_SET_TYPE_KERNEL_ARG(int, n)
			CL_SET_ARRAY_KERNEL_ARG(&data_im)
			CL_SET_TYPE_KERNEL_ARG(int, data_im_step)
			CL_SET_TYPE_KERNEL_ARG(int, height)
			CL_SET_TYPE_KERNEL_ARG(int, width)
			CL_SET_TYPE_KERNEL_ARG(int, kernel_h)
			CL_SET_TYPE_KERNEL_ARG(int, kernel_w)
			CL_SET_TYPE_KERNEL_ARG(int, pad_h)
			CL_SET_TYPE_KERNEL_ARG(int, pad_w)
			CL_SET_TYPE_KERNEL_ARG(int, stride_h)
			CL_SET_TYPE_KERNEL_ARG(int, stride_w)
			CL_SET_TYPE_KERNEL_ARG(int, height_col)
			CL_SET_TYPE_KERNEL_ARG(int, width_col)
			CL_SET_ARRAY_KERNEL_ARG(&data_col)
			CL_SET_TYPE_KERNEL_ARG(int, data_col_step)

			size_t global 	= CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
			size_t local 	= CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

			err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
			if (err != CL_SUCCESS) {
				LOG(ERROR)<< "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
				return false;
			}
			clFinish(*queue);
			LOG(INFO)<< "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

			CL_SET_KERNEL_ARG_END

			return true;
		}
		template bool clim2col_gpu<float>(const int n, const float* data_im, const int data_im_step, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, float* data_col, const int data_col_step);
		template bool clim2col_gpu<double>(const int n, const double* data_im, const int data_im_step, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h,	const int stride_w, const int height_col, const int width_col, double* data_col, const int data_col_step);

		template<typename T> bool clim2col_gpu(const int num_kernels, const T* data_im, const int bottom_step, const int num_images, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w,	const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, T* data_col, const int top_step) {

			std::string kernel_name = clGetKernelName<T>("clim2col_perf");

			queue = gpu->getQueue();
			if (!queue) {
				LOG(ERROR)<< gpu->name() << "> failed to get OpenCL command queue";
				return false;
			}

			kernel = gpu->getKernel(kernel_name);
			if (kernel == NULL) {
				return false;
			}

			CL_SET_KERNEL_ARG
			CL_SET_TYPE_KERNEL_ARG(int, num_kernels)
			CL_SET_ARRAY_KERNEL_ARG(&data_im)
			CL_SET_TYPE_KERNEL_ARG(int, bottom_step)
			CL_SET_TYPE_KERNEL_ARG(int, num_channels)
			CL_SET_TYPE_KERNEL_ARG(int, height)
			CL_SET_TYPE_KERNEL_ARG(int, width)
			CL_SET_TYPE_KERNEL_ARG(int, kernel_h)
			CL_SET_TYPE_KERNEL_ARG(int, kernel_w)
			CL_SET_TYPE_KERNEL_ARG(int, pad_h)
			CL_SET_TYPE_KERNEL_ARG(int, pad_w)
			CL_SET_TYPE_KERNEL_ARG(int, stride_h)
			CL_SET_TYPE_KERNEL_ARG(int, stride_w)
			CL_SET_TYPE_KERNEL_ARG(int, height_col)
			CL_SET_TYPE_KERNEL_ARG(int, width_col)
			CL_SET_ARRAY_KERNEL_ARG(&data_col)
			CL_SET_TYPE_KERNEL_ARG(int, top_step)

			int dim = 1;
			size_t *global;
			size_t *local;

			size_t global3D[3] 	= { CAFFE_GET_GLOBAL_WORKITEMS(width_col, OPENCL_LOCAL_SIZE), (size_t) height_col, (size_t) num_images * num_channels };
			size_t local3D[3] 	= { OPENCL_LOCAL_SIZE, 1, 1 };
			size_t global1D 		= CAFFE_GET_GLOBAL_WORKITEMS(num_kernels, OPENCL_LOCAL_SIZE);
			size_t local1D 		= CAFFE_GET_LOCAL_WORKITEMS(num_kernels, OPENCL_LOCAL_SIZE);

			int use = 1;
			switch (use) {
				case 1: {
					dim = 3;
					global = &global3D[0];
					local = &local3D[0];
				}
					break;
				default: {
					dim = 1;
					global = &global1D;
					local = &local1D;
				}
					break;
			}

			err = clEnqueueNDRangeKernel(*queue, *kernel, dim, NULL, global, local, 0, NULL, NULL);
			if (err != CL_SUCCESS) {
				LOG(ERROR)<< "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
				return false;
			}
			clFinish(*queue);
			LOG(INFO)<< "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

			CL_SET_KERNEL_ARG_END

			return true;
		}
		template bool clim2col_gpu<float>(const int num_kernels, const float* data_im, const int bottom_step, const int num_images, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, float* data_col, const int top_step);
		template bool clim2col_gpu<double>(const int num_kernels, const double* data_im, const int bottom_step, const int num_images, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, double* data_col, const int top_step);

		template<typename T> bool clcol2im_gpu(const int n, const T* data_col, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, T* data_im) {

			std::string kernel_name = clGetKernelName<T>("clcol2im");

			queue = gpu->getQueue();
			if (!queue) {
				LOG(ERROR)<< gpu->name() << "> failed to get OpenCL command queue";
				return false;
			}

			kernel = gpu->getKernel(kernel_name);
			if (kernel == NULL) {
				return false;
			}

			CL_SET_KERNEL_ARG
			CL_SET_TYPE_KERNEL_ARG(int, n)
			CL_SET_ARRAY_KERNEL_ARG(&data_col)
			CL_SET_TYPE_KERNEL_ARG(int, height)
			CL_SET_TYPE_KERNEL_ARG(int, width)
			CL_SET_TYPE_KERNEL_ARG(int, channels)
			CL_SET_TYPE_KERNEL_ARG(int, patch_h)
			CL_SET_TYPE_KERNEL_ARG(int, patch_w)
			CL_SET_TYPE_KERNEL_ARG(int, pad_h)
			CL_SET_TYPE_KERNEL_ARG(int, pad_w)
			CL_SET_TYPE_KERNEL_ARG(int, stride_h)
			CL_SET_TYPE_KERNEL_ARG(int, stride_w)
			CL_SET_TYPE_KERNEL_ARG(int, height_col)
			CL_SET_TYPE_KERNEL_ARG(int, width_col)
			CL_SET_ARRAY_KERNEL_ARG(&data_im)

			size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
			size_t local = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

			err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
			if (err != CL_SUCCESS) {
				LOG(ERROR)<< "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
				return false;
			}
			clFinish(*queue);
			LOG(INFO)<< "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

			CL_SET_KERNEL_ARG_END

			return true;
		}

		template bool clcol2im_gpu<float>(const int n, const float* data_col, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, float* data_im);
		template bool clcol2im_gpu<double>(const int n, const double* data_col, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, double* data_im);

		template<typename T> bool clcol2im_gpu(const int n, const T* data_col, const int data_col_step, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, T* data_im, const int data_im_step) {

			std::string kernel_name = clGetKernelName<T>("clcol2im_perf2");

			queue = gpu->getQueue();
			if (!queue) {
				LOG(ERROR)<< gpu->name() << "> failed to get OpenCL command queue";
				return false;
			}

			kernel = gpu->getKernel(kernel_name);
			if (kernel == NULL) {
				return false;
			}

			CL_SET_KERNEL_ARG
			CL_SET_TYPE_KERNEL_ARG(int, n)
			CL_SET_ARRAY_KERNEL_ARG(&data_col)
			CL_SET_TYPE_KERNEL_ARG(int, data_col_step)
			CL_SET_TYPE_KERNEL_ARG(int, height)
			CL_SET_TYPE_KERNEL_ARG(int, width)
			CL_SET_TYPE_KERNEL_ARG(int, channels)
			CL_SET_TYPE_KERNEL_ARG(int, patch_h)
			CL_SET_TYPE_KERNEL_ARG(int, patch_w)
			CL_SET_TYPE_KERNEL_ARG(int, pad_h)
			CL_SET_TYPE_KERNEL_ARG(int, pad_w)
			CL_SET_TYPE_KERNEL_ARG(int, stride_h)
			CL_SET_TYPE_KERNEL_ARG(int, stride_w)
			CL_SET_TYPE_KERNEL_ARG(int, height_col)
			CL_SET_TYPE_KERNEL_ARG(int, width_col)
			CL_SET_ARRAY_KERNEL_ARG(&data_im)
			CL_SET_TYPE_KERNEL_ARG(int, data_im_step)

			size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
			size_t local = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

			err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
			if (err != CL_SUCCESS) {
				LOG(ERROR)<< "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
				return false;
			}
			clFinish(*queue);
			LOG(INFO)<< "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

			CL_SET_KERNEL_ARG_END

			return true;
		}
		template bool clcol2im_gpu<float>(const int n, const float* data_col, const int data_col_step, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, float* data_im, const int data_im_step);
		template bool clcol2im_gpu<double>(const int n, const double* data_col, const int data_col_step, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w,	const int stride_h, const int stride_w, const int height_col, const int width_col, double* data_im, const int data_im_step);

		template<typename T> bool clcol2im_gpu(const int n, const T* data_col, const int top_step, const int col_number, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, T* data_im, const int bottom_step) {

			std::string kernel_name = clGetKernelName<T>("clcol2im_perf");

			queue = gpu->getQueue();
			if (!queue) {
				LOG(ERROR)<< gpu->name() << "> failed to get OpenCL command queue";
				return false;
			}

			kernel = gpu->getKernel(kernel_name);
			if (kernel == NULL) {
				return false;
			}

			CL_SET_KERNEL_ARG
			CL_SET_TYPE_KERNEL_ARG(int, n)
			CL_SET_ARRAY_KERNEL_ARG(&data_col)
			CL_SET_TYPE_KERNEL_ARG(int, top_step)
			CL_SET_TYPE_KERNEL_ARG(int, col_number)
			CL_SET_TYPE_KERNEL_ARG(int, height)
			CL_SET_TYPE_KERNEL_ARG(int, width)
			CL_SET_TYPE_KERNEL_ARG(int, channels)
			CL_SET_TYPE_KERNEL_ARG(int, patch_h)
			CL_SET_TYPE_KERNEL_ARG(int, patch_w)
			CL_SET_TYPE_KERNEL_ARG(int, pad_h)
			CL_SET_TYPE_KERNEL_ARG(int, pad_w)
			CL_SET_TYPE_KERNEL_ARG(int, stride_h)
			CL_SET_TYPE_KERNEL_ARG(int, stride_w)
			CL_SET_TYPE_KERNEL_ARG(int, height_col)
			CL_SET_TYPE_KERNEL_ARG(int, width_col)
			CL_SET_ARRAY_KERNEL_ARG(&data_im)
			CL_SET_TYPE_KERNEL_ARG(int, bottom_step)

			size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
			size_t local = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

			err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
			if (err != CL_SUCCESS) {
				LOG(ERROR)<< "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
				return false;
			}
			clFinish(*queue);
			LOG(INFO)<< "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

			CL_SET_KERNEL_ARG_END

			return true;
		}
		template bool clcol2im_gpu<float>(const int n, const float* data_col, const int top_step, const int col_number, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, float* data_im, const int bottom_step);
		template bool clcol2im_gpu<double>(const int n, const double* data_col, const int top_step, const int col_number, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h,	const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, double* data_im, const int bottom_step);

	} // namespace OpenCL

	template<typename Dtype>
	void im2col_gpu(const Dtype* data_im, const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* data_col) {
		// We are going to launch channels * height_col * width_col kernels, each
		// kernel responsible for copying a single-channel grid.
		int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
		int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
		int num_kernels = channels * height_col * width_col;

		/*
		 // NOLINT_NEXT_LINE(whitespace/operators)
		 im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
		 CAFFE_CUDA_NUM_THREADS>>>(
		 num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
		 pad_w, stride_h, stride_w, height_col,
		 width_col, data_col);
		 CUDA_POST_KERNEL_CHECK;
		 */
		BOOL_CHECK(caffe::OpenCL::clim2col_gpu(num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, height_col, width_col, data_col));
	}
	template void im2col_gpu<float>(const float* data_im, const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, float* data_col);
	template void im2col_gpu<double>(const double* data_im, const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, double* data_col);

	template<typename Dtype>
	void im2col_gpu(const Dtype* data_im, const int data_im_step, const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* data_col, const int data_col_step) {
		// We are going to launch channels * height_col * width_col kernels, each
		// kernel responsible for copying a single-channel grid.
		int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
		int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
		int num_kernels = channels * height_col * width_col;

		/*
		 // NOLINT_NEXT_LINE(whitespace/operators)
		 im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
		 CAFFE_CUDA_NUM_THREADS>>>(
		 num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
		 pad_w, stride_h, stride_w, height_col,
		 width_col, data_col);
		 CUDA_POST_KERNEL_CHECK;
		 */
		BOOL_CHECK(caffe::OpenCL::clim2col_gpu(num_kernels, data_im, data_im_step, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, height_col, width_col, data_col, data_col_step));
	}
	template void im2col_gpu<float>(const float* data_im, const int data_im_step, const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, float* data_col, const int data_col_step);
	template void im2col_gpu<double>(const double* data_im, const int data_im_step, const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, double* data_col, const int data_col_step);

	template<typename Dtype>
	void im2col_gpu(const Dtype* data_im, const int bottom_step, const int num_images, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h,	const int stride_w, Dtype* data_col, const int top_step) {

		// We are going to launch channels * height_col * width_col kernels, each
		// kernel responsible for copying a single-channel grid.
		int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
		int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
		int num_kernels = num_images * num_channels * height_col * width_col;

		BOOL_CHECK(caffe::OpenCL::clim2col_gpu(num_kernels, data_im, bottom_step, num_images, num_channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, height_col, width_col, data_col, top_step));
	}
	template void im2col_gpu<float>(const float* data_im, const int bottom_step, const int num_images, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
			const int stride_h, const int stride_w, float* data_col, const int top_step);
	template void im2col_gpu<double>(const double* data_im, const int bottom_step, const int num_images, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
			const int stride_h, const int stride_w, double* data_col, const int top_step);

	template<typename Dtype>
	void col2im_gpu(const Dtype* data_col, const int channels, const int height, const int width, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* data_im) {
		int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
		int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
		int num_kernels = channels * height * width;
		// To avoid involving atomic operations, we will launch one kernel per
		// bottom dimension, and then in the kernel add up the top dimensions.
		/*
		 // NOLINT_NEXT_LINE(whitespace/operators)
		 col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
		 CAFFE_CUDA_NUM_THREADS>>>(
		 num_kernels, data_col, height, width, channels, patch_h, patch_w,
		 pad_h, pad_w, stride_h, stride_w,
		 height_col, width_col, data_im);
		 CUDA_POST_KERNEL_CHECK;
		 */
		BOOL_CHECK(caffe::OpenCL::clcol2im_gpu(num_kernels, data_col, height, width, channels, patch_h, patch_w, pad_h, pad_w, stride_h, stride_w, height_col, width_col, data_im));

	}
	template void col2im_gpu<float>(const float* data_col, const int channels, const int height, const int width, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, float* data_im);
	template void col2im_gpu<double>(const double* data_col, const int channels, const int height, const int width, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,	double* data_im);

	template<typename Dtype>
	void col2im_gpu(const Dtype* data_col, const int data_col_step, const int channels, const int height, const int width, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* data_im, const int data_im_step) {
		int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
		int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
		int num_kernels = channels * height * width;
		// To avoid involving atomic operations, we will launch one kernel per
		// bottom dimension, and then in the kernel add up the top dimensions.
		/*
		 // NOLINT_NEXT_LINE(whitespace/operators)
		 col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
		 CAFFE_CUDA_NUM_THREADS>>>(
		 num_kernels, data_col, height, width, channels, patch_h, patch_w,
		 pad_h, pad_w, stride_h, stride_w,
		 height_col, width_col, data_im);
		 CUDA_POST_KERNEL_CHECK;
		 */
		BOOL_CHECK(caffe::OpenCL::clcol2im_gpu(num_kernels, data_col, data_col_step, height, width, channels, patch_h, patch_w, pad_h, pad_w, stride_h, stride_w, height_col, width_col, data_im, data_im_step));

	}
	template void col2im_gpu<float>(const float* data_col, const int data_col_step, const int channels, const int height, const int width, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, float* data_im, const int data_im_step);
	template void col2im_gpu<double>(const double* data_col, const int data_col_step, const int channels, const int height, const int width, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, double* data_im, const int data_im_step);

	template<typename Dtype>
	void col2im_gpu(const Dtype* data_col, const int top_step, const int col_number, const int channels, const int height, const int width, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* data_im, const int bottom_step) {

		int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
		int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
		int num_kernels = col_number * channels * height * width;
		// To avoid involving atomic operations, we will launch one kernel per
		// bottom dimension, and then in the kernel add up the top dimensions.
		/*
		 // NOLINT_NEXT_LINE(whitespace/operators)
		 col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
		 CAFFE_CUDA_NUM_THREADS>>>(
		 num_kernels, data_col, height, width, channels, patch_h, patch_w,
		 pad_h, pad_w, stride_h, stride_w,
		 height_col, width_col, data_im);
		 CUDA_POST_KERNEL_CHECK;
		 */
		BOOL_CHECK(caffe::OpenCL::clcol2im_gpu(num_kernels, data_col, top_step, col_number, height, width, channels, patch_h, patch_w, pad_h, pad_w, stride_h, stride_w, height_col, width_col, data_im, bottom_step));

	}
	template void col2im_gpu<float>(const float* data_col, const int top_step, const int col_number, const int channels, const int height, const int width, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, float* data_im, const int bottom_step);
	template void col2im_gpu<double>(const double* data_col, const int top_step, const int col_number, const int channels, const int height, const int width, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, double* data_im, const int bottom_step);

#endif // USE_OPENCL

}  // namespace caffe
