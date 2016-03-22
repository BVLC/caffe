#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "caffe/common.hpp"
#ifndef CPU_ONLY
#if defined(USE_GREENTEA) && defined(USE_FFT)
#include "caffe/device.hpp"
#include "caffe/greentea/cl_kernels.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#include "caffe/util/fft.hpp"

// #define DEBUG_PROFILE

namespace caffe {

#ifdef DEBUG_PROFILE
void kernel_execution_time(cl_event* event, const char* kernel_name) {
  cl_ulong time_start, time_end;
  clWaitForEvents(1, event);
  clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START,
      sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(time_end),
      &time_end, NULL);
  clReleaseEvent(*event);
  std::cout << "* Execution time (" << kernel_name << ") = " <<
      ((time_end - time_start) / 1000000.0) << " ms." << std::endl;
}
#endif

void clear_gpu_fft_buffer(void* data, const int size) {
  device *dc = Caffe::GetDefaultDevice();
  greentea_memset(dc->id(), size, 0, (cl_mem) data, 0);
}

// Copy and cyclic-shift 0 padding of weights to FFT real buffer
template <typename Dtype>
void fft_gpu_copy2buffer(Dtype* fft_gpu_weights_real, const Dtype* weight,
    int num_output, int group, int channels, int ker_h, int ker_w,
    int ker_c_h, int ker_c_w, int fft_height, int fft_width) {
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();


  // size_t aligned_offset_fft_gpu_weights_real;
  int offset_offset_fft_gpu_weights_real = 0;

  int offset_offset_weight = 0;

  const int ch_gr = channels / group;
  const int ker_size_ch_group = ker_h * ker_w * ch_gr;
  const size_t global_work_size = num_output * ker_size_ch_group;
  int argIdx = 0;
  const int ker_size = ker_h * ker_w;
  const int complex_width_len = 2*(fft_width/2 + 1);
  viennacl::ocl::kernel & kernel = ctx.get_kernel("kernel_program",
    CL_KERNEL_SELECT("copy2buffer_cyclic_shift_in"));
  kernel.arg(argIdx++, WrapHandle((cl_mem)fft_gpu_weights_real, &ctx));
  kernel.arg(argIdx++, offset_offset_fft_gpu_weights_real);
  kernel.arg(argIdx++, WrapHandle((cl_mem)weight, &ctx));
  kernel.arg(argIdx++, offset_offset_weight);
  kernel.arg(argIdx++, ker_size);
  kernel.arg(argIdx++, ch_gr);
  kernel.arg(argIdx++, ker_size_ch_group);
  kernel.arg(argIdx++, ker_w);
  kernel.arg(argIdx++, ker_c_h);
  kernel.arg(argIdx++, ker_c_w);
  kernel.arg(argIdx++, fft_height);
  kernel.arg(argIdx++, fft_width);
  kernel.arg(argIdx++, complex_width_len);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(), kernel, 1,
            NULL, &global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "copy2buffer_cyclic_shift_in");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
            kernel.handle().get(), 1, NULL, &global_work_size, NULL,
            0, NULL, NULL));
#endif
}
template void fft_gpu_copy2buffer<float>(float* fft_gpu_weights_real,
    const float* weight, int num_output, int group, int channels,
    int ker_h, int ker_w, int ker_c_h, int ker_c_w,
    int fft_height, int fft_width);
template void fft_gpu_copy2buffer<double>(double* fft_gpu_weights_real,
    const double* weight, int num_output, int group,
    int channels, int ker_h, int ker_w, int ker_c_h, int ker_c_w,
    int fft_height, int fft_width);

// Copy and left-top 0 padding of data to FFT real buffer
template <typename Dtype>
void fft_gpu_copy2buffer_in_2D(Dtype* map_out, const Dtype* map_in,
    int in_offset, int channels, int height_out, int width_out,
    int height, int width, int stride_h, int stride_w, int pad_h,
    int pad_w) {
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();


  int offset_offset_map_out = 0;
  int offset_offset_map_in = in_offset;

  int map_out_size = height_out * width_out;
  int size = height * width;
  int count = size >> 2;
  const size_t global_work_size[2] = { (size_t)size, (size_t)channels };
  viennacl::ocl::kernel kernel;
  if (width < 4) {
    kernel = ctx.get_kernel("kernel_program",
      CL_KERNEL_SELECT("copy2buffer_left_top_in_naive_2d"));
  } else {
    kernel = ctx.get_kernel("kernel_program",
      CL_KERNEL_SELECT("copy2buffer_left_top_in_2d"));
  }
  int argIdx = 0;
  kernel.arg(argIdx++, WrapHandle((cl_mem)map_out, &ctx));
  kernel.arg(argIdx++, offset_offset_map_out);
  kernel.arg(argIdx++, WrapHandle((cl_mem)map_in, &ctx));
  kernel.arg(argIdx++, offset_offset_map_in);
  kernel.arg(argIdx++, map_out_size);
  kernel.arg(argIdx++, size);
  kernel.arg(argIdx++, count);
  kernel.arg(argIdx++, height_out);
  kernel.arg(argIdx++, width_out);
  kernel.arg(argIdx++, height);
  kernel.arg(argIdx++, width);
  kernel.arg(argIdx++, stride_h);
  kernel.arg(argIdx++, stride_w);
  kernel.arg(argIdx++, pad_h);
  kernel.arg(argIdx++, pad_w);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(), kernel, 2,
      NULL, global_work_size, NULL, 0, NULL, &event));
  if (width < 4)
    kernel_execution_time(&event, "copy2buffer_left_top_in_naive_2d");
  else
    kernel_execution_time(&event, "copy2buffer_left_top_in_2d");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
      kernel.handle().get(), 2, NULL, global_work_size, NULL, 0, NULL, NULL));
#endif
}
template void fft_gpu_copy2buffer_in_2D<float>(float* map_out,
    const float* map_in, int in_offset, int channels,
    int height_out, int width_out, int height, int width,
    int stride_h, int stride_w, int pad_h, int pad_w);
template void fft_gpu_copy2buffer_in_2D<double>(double* map_out,
    const double* map_in, int in_offset, int channels,
    int height_out, int width_out, int height, int width,
    int stride_h, int stride_w, int pad_h, int pad_w);

// Copy from left-top 0 padded data to real buffer
template <typename Dtype>
void fft_gpu_copy2buffer_out_forward_2D(Dtype* map_out, int out_offset,
    const Dtype* map_in, int num_output, int height_out, int width_out,
    int fft_height, int fft_width, int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();
  int offset_offset_map_out = out_offset;
  int offset_offset_map_in = 0;

  int size = height_out * width_out;
  int count = size >> 2;
  int map_in_size = fft_height * fft_width;
  const size_t global_work_size[2] = { (size_t)size, (size_t)num_output };
  viennacl::ocl::kernel kernel;
  if (width_out < 4) {
    kernel = ctx.get_kernel("kernel_program",
        CL_KERNEL_SELECT("copy2buffer_left_top_out_naive_2d"));
  } else {
    kernel = ctx.get_kernel("kernel_program",
        CL_KERNEL_SELECT("copy2buffer_left_top_out_2d"));
  }
  int argIdx = 0;
  kernel.arg(argIdx++, WrapHandle((cl_mem)map_out, &ctx));
  kernel.arg(argIdx++, offset_offset_map_out);
  kernel.arg(argIdx++, WrapHandle((cl_mem)map_in, &ctx));
  kernel.arg(argIdx++, offset_offset_map_in);
  kernel.arg(argIdx++, size);
  kernel.arg(argIdx++, count);
  kernel.arg(argIdx++, map_in_size);
  kernel.arg(argIdx++, height_out);
  kernel.arg(argIdx++, width_out);
  kernel.arg(argIdx++, fft_height);
  kernel.arg(argIdx++, fft_width);
  kernel.arg(argIdx++, kernel_center_h);
  kernel.arg(argIdx++, kernel_center_w);
  kernel.arg(argIdx++, stride_h);
  kernel.arg(argIdx++, stride_w);
  kernel.arg(argIdx++, pad_h);
  kernel.arg(argIdx++, pad_w);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(), kernel, 2,
      NULL, global_work_size, NULL, 0, NULL, &event));
  if (width_out < 4)
    kernel_execution_time(&event, "copy2buffer_left_top_out_naive_2d");
  else
    kernel_execution_time(&event, "copy2buffer_left_top_out_2d");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
      kernel.handle().get(), 2, NULL, global_work_size, NULL, 0, NULL, NULL));
#endif
}
template void fft_gpu_copy2buffer_out_forward_2D<float>(float* map_out,
    int out_offset, const float* map_in, int num_output,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w);
template void fft_gpu_copy2buffer_out_forward_2D<double>(double* map_out,
    int out_offset,  const double* map_in, int num_output,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w);

template <typename Dtype>
void fft_gpu_copy2buffer_out_backward(Dtype* map_out, const Dtype* map_in,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();

  int offset_offset_map_out = 0;
  int offset_offset_map_in = 0;

  const size_t global_work_size = height_out * width_out;
  viennacl::ocl::kernel &kernel = ctx.get_kernel("kernel_program",
      CL_KERNEL_SELECT("copy2buffer_cyclic_shift_out"));
  int argIdx = 0;
  kernel.arg(argIdx++, WrapHandle((cl_mem)map_out, &ctx));
  kernel.arg(argIdx++, offset_offset_map_out);
  kernel.arg(argIdx++, WrapHandle((cl_mem)map_in, &ctx));
  kernel.arg(argIdx++, offset_offset_map_in);
  kernel.arg(argIdx++, width_out);
  kernel.arg(argIdx++, fft_height);
  kernel.arg(argIdx++, fft_width);
  kernel.arg(argIdx++, kernel_center_h);
  kernel.arg(argIdx++, kernel_center_w);
  kernel.arg(argIdx++, stride_h);
  kernel.arg(argIdx++, stride_w);
  kernel.arg(argIdx++, pad_h);
  kernel.arg(argIdx++, pad_w);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(), kernel, 1,
      NULL, &global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "copy2buffer_cyclic_shift_out");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
      kernel.handle().get(), 1, NULL, &global_work_size, NULL, 0,
      NULL, NULL));
#endif
}
template void fft_gpu_copy2buffer_out_backward<float>(float* map_out,
    const float* map_in,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w);
template void fft_gpu_copy2buffer_out_backward<double>(double* map_out,
    const double* map_in,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w);

template <typename Dtype>
void fft_gpu_copy2buffer_out_backward_2D(Dtype* map_out, int out_offset,
    const Dtype* map_in, int channels, int height_out, int width_out,
    int fft_height, int fft_width, int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();

  int offset_offset_map_out = out_offset;
  int offset_offset_map_in = 0;
  int map_out_size = height_out * width_out;
  int map_in_size = fft_height * fft_width;
  const size_t global_work_size[2] = { (size_t)map_out_size,
                                       (size_t)channels };
  viennacl::ocl::kernel &kernel = ctx.get_kernel("kernel_program",
      CL_KERNEL_SELECT("copy2buffer_cyclic_shift_out_2d"));
  int argIdx = 0;
  kernel.arg(argIdx++, WrapHandle((cl_mem)map_out, &ctx));
  kernel.arg(argIdx++, offset_offset_map_out);
  kernel.arg(argIdx++, WrapHandle((cl_mem)map_in, &ctx));
  kernel.arg(argIdx++, offset_offset_map_in);
  kernel.arg(argIdx++, map_out_size);
  kernel.arg(argIdx++, map_in_size);
  kernel.arg(argIdx++, width_out);
  kernel.arg(argIdx++, fft_height);
  kernel.arg(argIdx++, fft_width);
  kernel.arg(argIdx++, kernel_center_h);
  kernel.arg(argIdx++, kernel_center_w);
  kernel.arg(argIdx++, stride_h);
  kernel.arg(argIdx++, stride_w);
  kernel.arg(argIdx++, pad_h);
  kernel.arg(argIdx++, pad_w);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(), kernel, 2,
      NULL, global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "copy2buffer_cyclic_shift_out_2d");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
      kernel.handle().get(), 2, NULL, global_work_size, NULL, 0,
      NULL, NULL));
#endif
}
template void fft_gpu_copy2buffer_out_backward_2D<float>(float* map_out,
    int out_offset, const float* map_in, int channels,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w);
template void fft_gpu_copy2buffer_out_backward_2D<double>(double* map_out,
    int out_offset, const double* map_in, int channels,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w);

template <typename Dtype>
void caffe_gpu_elementMulConj_1D(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int map_size, const int ch_gr) {
  // Note: map_size is the number of DtypeComplex values
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();
  int offset_offset_dst = 0;
  int offset_offset_src1 = 0;
  int offset_offset_src2 = 0;


  const size_t global_work_size = map_size >> 1;
  viennacl::ocl::kernel kernel = ctx.get_kernel("kernel_program",
      CL_KERNEL_SELECT("complex_conjugate_multiplication_1d"));
  int argIdx = 0;
  kernel.arg(argIdx++, WrapHandle((cl_mem)dst, &ctx));
  kernel.arg(argIdx++, offset_offset_dst << 1);
  kernel.arg(argIdx++, WrapHandle((cl_mem)src1, &ctx));
  kernel.arg(argIdx++, offset_offset_src1 << 1);
  kernel.arg(argIdx++, WrapHandle((cl_mem)src2, &ctx));
  kernel.arg(argIdx++, offset_offset_src2 << 1);
  kernel.arg(argIdx++, ch_gr);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(), kernel, 1,
      NULL, &global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "complex_conjugate_multiplication_1d");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
      kernel.handle().get(), 1, NULL, &global_work_size, NULL, 0,
      NULL, NULL));
#endif
}
template void caffe_gpu_elementMulConj_1D<float>(DtypeComplex<float>* dst,
    const DtypeComplex<float>* src1, const DtypeComplex<float>* src2,
    const int map_size, const int ch_gr);
template void caffe_gpu_elementMulConj_1D<double>(DtypeComplex<double>* dst,
    const DtypeComplex<double>* src1, const DtypeComplex<double>* src2,
    const int map_size, const int ch_gr);

template <typename Dtype>
void caffe_gpu_elementMulConj_Reshape(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int out_gr, const int map_size, const int ch_gr) {
  // Note: map_size is the number of DtypeComplex values
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();

  cl_command_queue queue = ctx.get_queue().handle().get();
  size_t block_size = map_size * ch_gr * sizeof(DtypeComplex<Dtype>);
  cl_mem src1_vec = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      block_size, NULL, NULL);
  size_t global_work_size1 = map_size;
  viennacl::ocl::kernel kernel = ctx.get_kernel("kernel_program",
      CL_KERNEL_SELECT("convert_data_to_channel_major"));
  int argIdx = 0;
  kernel.arg(argIdx++, WrapHandle((cl_mem)src1_vec, &ctx));
  kernel.arg(argIdx++, WrapHandle((cl_mem)src1, &ctx));
  kernel.arg(argIdx++, map_size);
  kernel.arg(argIdx++, ch_gr);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
      &global_work_size1, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "Reshape data to channel major");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel.handle().get(), 1, NULL,
      &global_work_size1, NULL, 0, NULL, NULL));
#endif

  viennacl::ocl::kernel kernel_batchedCdotc = ctx.get_kernel("kernel_program",
      CL_KERNEL_SELECT("batchedCdotc"));
  // Batched complex number dot product
  size_t global_work_size2[2] = { (size_t)map_size, (size_t)out_gr };
  argIdx = 0;
  kernel_batchedCdotc.arg(argIdx++, WrapHandle((cl_mem)dst, &ctx));
  kernel_batchedCdotc.arg(argIdx++, WrapHandle((cl_mem)src1_vec, &ctx));
  kernel_batchedCdotc.arg(argIdx++, WrapHandle((cl_mem)src2, &ctx));
  kernel_batchedCdotc.arg(argIdx++, map_size);
  kernel_batchedCdotc.arg(argIdx++, ch_gr);
  kernel_batchedCdotc.arg(argIdx++, out_gr);
#ifdef DEBUG_PROFILE
  OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel_batchedCdotc, 2, NULL,
      global_work_size2, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "Batched complex dot product");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel_batchedCdotc.handle().get(), 2,
      NULL, global_work_size2, NULL, 0, NULL, NULL));
#endif
  clReleaseMemObject(src1_vec);
}
template void caffe_gpu_elementMulConj_Reshape<float>(
    DtypeComplex<float>* dst,
    const DtypeComplex<float>* src1, const DtypeComplex<float>* src2,
    const int out_gr, const int map_size, const int ch_gr);
template void caffe_gpu_elementMulConj_Reshape<double>(
    DtypeComplex<double>* dst,
    const DtypeComplex<double>* src1, const DtypeComplex<double>* src2,
    const int out_gr, const int map_size, const int ch_gr);

template <typename Dtype>
void caffe_gpu_elementMulConj_2D(DtypeComplex<Dtype>* dst, int dst_offset,
    const DtypeComplex<Dtype>* src1, int src1_offset,
    const DtypeComplex<Dtype>* src2, int src2_offset,
    const int out_gr, const int map_size, const int ch_gr) {
  // Note: map_size is the number of DtypeComplex values
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();
  int offset_offset_dst = dst_offset;
  int offset_offset_src1 = src1_offset;
  int offset_offset_src2 = src2_offset;


  const size_t global_work_size[2] = { (size_t)map_size >> 1, (size_t)out_gr };
  viennacl::ocl::kernel kernel = ctx.get_kernel("kernel_program",
      CL_KERNEL_SELECT("complex_conjugate_multiplication_2d"));
  int argIdx = 0;
  kernel.arg(argIdx++, WrapHandle((cl_mem)dst, &ctx));
  kernel.arg(argIdx++, offset_offset_dst << 1);
  kernel.arg(argIdx++, WrapHandle((cl_mem)src1, &ctx));
  kernel.arg(argIdx++, offset_offset_src1 << 1);
  kernel.arg(argIdx++, WrapHandle((cl_mem)src2, &ctx));
  kernel.arg(argIdx++, offset_offset_src2 << 1);
  kernel.arg(argIdx++, out_gr);
  kernel.arg(argIdx++, map_size >> 1);
  kernel.arg(argIdx++, ch_gr);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(), kernel, 2,
      NULL, global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "complex_conjugate_multiplication_2d");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
      kernel.handle().get(), 2, NULL, global_work_size, NULL, 0, NULL, NULL));
#endif
}
template void caffe_gpu_elementMulConj_2D<float>(DtypeComplex<float>* dst,
    int dst_offset, const DtypeComplex<float>* src1, int src1_offset,
    const DtypeComplex<float>* src2, int src2_offset,
    const int out_gr, const int map_size, const int ch_gr);
template void caffe_gpu_elementMulConj_2D<double>(DtypeComplex<double>* dst,
    int dst_offset, const DtypeComplex<double>* src1, int src1_offset,
    const DtypeComplex<double>* src2, int src2_offset,
    const int out_gr, const int map_size, const int ch_gr);

template <typename Dtype>
void caffe_gpu_elementMulConj_2D_SLM(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int out_gr, const int map_size, const int ch_gr) {
  // Note: size is the number of DtypeComplex values
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();

  int offset_offset_dst = 0;
  int offset_offset_src1 = 0;
  int offset_offset_src2 = 0;


  int map_float4_size = map_size >> 1;
  // Note:
  // (16, 1) is good for Unit Test
  // (32, 16) is good for CaffNet
  // (128, 4) is perf hint recommended
  int local_work_size_x = (map_float4_size < 512) ? 16 : 32;  // TODO: Temporary
  int local_work_size_y = (out_gr < 16) ? 1 : 16;  // TODO: Temporary
  /*TODO: Temporary comment out
  if (out_gr >=  16 &&
      state.get_properties().device_max_work_group_size < 512) {
    local_work_size_y = 8;
  }*/
  const size_t local_work_size[2] = { (size_t)local_work_size_x,
                                      (size_t)local_work_size_y };
  int global_work_size_x =
      CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(map_float4_size, local_work_size_x);
  int global_work_size_y =
      CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(out_gr, local_work_size_y);
  const size_t global_work_size[2] = { (size_t)global_work_size_x,
                                       (size_t)global_work_size_y };
  viennacl::ocl::kernel kernel = ctx.get_kernel("kernel_program",
      CL_KERNEL_SELECT("complex_conjugate_multiplication_2d_SLM"));
  int argIdx = 0;
  kernel.arg(argIdx++, WrapHandle((cl_mem)dst, &ctx));
  kernel.arg(argIdx++, offset_offset_dst << 1);
  kernel.arg(argIdx++, WrapHandle((cl_mem)src1, &ctx));
  kernel.arg(argIdx++, offset_offset_src1 << 1);
  kernel.arg(
      argIdx++, ch_gr * local_work_size_x * sizeof(Dtype) * 4);
  kernel.arg(argIdx++, WrapHandle((cl_mem)src2, &ctx));
  kernel.arg(argIdx++, offset_offset_src2 << 1);
  kernel.arg(argIdx++, out_gr);
  kernel.arg(argIdx++, map_float4_size);
  kernel.arg(argIdx++, ch_gr);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(), kernel, 2,
      NULL, global_work_size, local_work_size, 0, NULL, &event));
  kernel_execution_time(&event, "complex_conjugate_multiplication_2d_SLM");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
      kernel.handle().get(), 2, NULL, global_work_size, local_work_size,
      0, NULL, NULL));
#endif
}
template void caffe_gpu_elementMulConj_2D_SLM<float>(DtypeComplex<float>* dst,
    const DtypeComplex<float>* src1, const DtypeComplex<float>* src2,
    const int out_gr, const int map_size, const int ch_gr);
template void caffe_gpu_elementMulConj_2D_SLM<double>(DtypeComplex<double>* dst,
    const DtypeComplex<double>* src1, const DtypeComplex<double>* src2,
    const int out_gr, const int map_size, const int ch_gr);

template <typename Dtype>
void caffe_gpu_elementMulConj_3D(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int out_gr, const int map_size, const int ch_gr) {
  // Note: map_size is the number of DtypeComplex values
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();

  int offset_offset_dst = 0;
  int offset_offset_src1 = 0;
  int offset_offset_src2 = 0;

  const size_t global_work_size[3] = { (size_t)map_size >> 1, (size_t)out_gr,
                                       (size_t)ch_gr };
  viennacl::ocl::kernel kernel = ctx.get_kernel("kernel_program",
      CL_KERNEL_SELECT("complex_conjugate_multiplication_3d"));
  int argIdx = 0;
  kernel.arg(argIdx++, WrapHandle((cl_mem)dst, &ctx));
  kernel.arg(argIdx++, offset_offset_dst << 1);
  kernel.arg(argIdx++, WrapHandle((cl_mem)src1, &ctx));
  kernel.arg(argIdx++, offset_offset_src1 << 1);
  kernel.arg(argIdx++, WrapHandle((cl_mem)src2, &ctx));
  kernel.arg(argIdx++, offset_offset_src2 << 1);
  kernel.arg(argIdx++, out_gr);
  kernel.arg(argIdx++, map_size >> 1);
  kernel.arg(argIdx++, ch_gr);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(), kernel, 3,
      NULL, global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "complex_conjugate_multiplication_3d");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
      kernel.handle().get(), 3, NULL, global_work_size, NULL, 0,
      NULL, NULL));
#endif
}
template void caffe_gpu_elementMulConj_3D<float>(DtypeComplex<float>* dst,
    const DtypeComplex<float>* src1, const DtypeComplex<float>* src2,
    const int out_gr, const int map_size, const int ch_gr);
template void caffe_gpu_elementMulConj_3D<double>(DtypeComplex<double>* dst,
    const DtypeComplex<double>* src1, const DtypeComplex<double>* src2,
    const int out_gr, const int map_size, const int ch_gr);

template <typename Dtype>
void caffe_gpu_elementMulConj_3D_SLM(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int out_gr, const int map_size, const int ch_gr) {
  // Note: size is the number of DtypeComplex values
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();

  int offset_offset_dst = 0;
  int offset_offset_src1 = 0;
  int offset_offset_src2 = 0;

  int map_float4_size = map_size >> 1;
  // Note:
  // (16, 1) is good for Unit Test
  // (32, 2) is good for CaffNet
  // (128, 4) is perf hint recommended
  int local_work_size_x = (map_float4_size < 512) ? 16 : 32;  // TODO: Temporary
  int local_work_size_y = (out_gr < 16) ? 1 : 2;  // TODO: Temporary
  int local_work_size_z = 1;
  const size_t local_work_size[3] = {
      (size_t)local_work_size_x, (size_t)local_work_size_y,
      (size_t)local_work_size_z };
  int global_work_size_x =
      CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(map_float4_size, local_work_size_x);
  int global_work_size_y =
      CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(out_gr, local_work_size_y);
  int global_work_size_z =
      CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(ch_gr, local_work_size_z);
  const size_t global_work_size[3] = {
      (size_t)global_work_size_x, (size_t)global_work_size_y,
      (size_t)global_work_size_z };
  viennacl::ocl::kernel kernel = ctx.get_kernel("kernel_program",
      CL_KERNEL_SELECT("complex_conjugate_multiplication_3d_SLM"));
  int argIdx = 0;
  kernel.arg(argIdx++, WrapHandle((cl_mem)dst, &ctx));
  kernel.arg(argIdx++, offset_offset_dst << 1);
  kernel.arg(
      argIdx++, ch_gr * sizeof(Dtype) * 4);
  kernel.arg(argIdx++, WrapHandle((cl_mem)src1, &ctx));
  kernel.arg(argIdx++, offset_offset_src1 << 1);
  kernel.arg(
      argIdx++, ch_gr * local_work_size_x * sizeof(Dtype) * 4);
  kernel.arg(argIdx++, WrapHandle((cl_mem)src2, &ctx));
  kernel.arg(argIdx++, offset_offset_src2 << 1);
  kernel.arg(argIdx++, out_gr);
  kernel.arg(argIdx++, map_float4_size);
  kernel.arg(argIdx++, ch_gr);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(), kernel, 3,
      NULL, global_work_size, local_work_size, 0, NULL, &event));
  kernel_execution_time(&event, "complex_conjugate_multiplication_3d_SLM");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
      kernel.handle().get(), 3, NULL, global_work_size, local_work_size, 0,
      NULL, NULL));
#endif
}
template void caffe_gpu_elementMulConj_3D_SLM<float>(DtypeComplex<float>* dst,
    const DtypeComplex<float>* src1, const DtypeComplex<float>* src2,
    const int out_gr, const int map_size, const int ch_gr);
template void caffe_gpu_elementMulConj_3D_SLM<double>(DtypeComplex<double>* dst,
    const DtypeComplex<double>* src1, const DtypeComplex<double>* src2,
    const int out_gr, const int map_size, const int ch_gr);

template <typename Dtype>
void caffe_gpu_elementMul_1D(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int size, const int ch_gr) {
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();

  int offset_offset_dst = 0;
  int offset_offset_src1 = 0;
  int offset_offset_src2 = 0;

  const size_t global_work_size = size >> 1;  // # of Dtype4
  viennacl::ocl::kernel kernel = ctx.get_kernel("kernel_program",
      CL_KERNEL_SELECT("complex_multiplication_1d"));
  int argIdx = 0;
  kernel.arg(argIdx++, WrapHandle((cl_mem)dst, &ctx));
  kernel.arg(argIdx++, offset_offset_dst << 1);
  kernel.arg(argIdx++, WrapHandle((cl_mem)src1, &ctx));
  kernel.arg(argIdx++, offset_offset_src1 << 1);
  kernel.arg(argIdx++, WrapHandle((cl_mem)src2, &ctx));
  kernel.arg(argIdx++, offset_offset_src2 << 1);
  kernel.arg(argIdx++, size >> 1);
  kernel.arg(argIdx++, ch_gr);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(), kernel, 1,
      NULL, &global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "complex_multiplication_1d");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
      kernel.handle().get(), 1, NULL, &global_work_size, NULL, 0,
      NULL, NULL));
#endif
}
template void caffe_gpu_elementMul_1D<float>(DtypeComplex<float>* dst,
    const DtypeComplex<float>* src1, const DtypeComplex<float>* src2,
    const int size, const int ch_gr);
template void caffe_gpu_elementMul_1D<double>(DtypeComplex<double>* dst,
    const DtypeComplex<double>* src1, const DtypeComplex<double>* src2,
    const int size, const int ch_gr);

template <typename Dtype>
void caffe_gpu_elementMul_2D_SLM(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int size, const int ch_gr, const int num_output) {
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();

  int offset_offset_dst = 0;
  int offset_offset_src1 = 0;
  int offset_offset_src2 = 0;

  // (16,2)=6K, (8,4)=1.5K work for CaffeNet
  // (128, 4) is perf hint recommended
  int local_work_size_x = 16;  // TODO: what is the best number?
  int local_work_size_y = 2;   // TODO: what is the best number?
  const size_t local_work_size[2] = { (size_t)local_work_size_x,
      (size_t)local_work_size_y };

  int map_size_in_dtype4 = size >> 1;  // # of Dtype4
  int global_work_size_x =
      CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(map_size_in_dtype4, local_work_size_x);
  int global_work_size_y =
      CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(num_output, local_work_size_y);
  const size_t global_work_size[2] = { (size_t)global_work_size_x,
                                       (size_t)global_work_size_y };
  const size_t local_mem_size_in_bytes =
      ch_gr * local_work_size_x * local_work_size_y * sizeof(Dtype) * 4;

  viennacl::ocl::kernel kernel = ctx.get_kernel("kernel_program",
      CL_KERNEL_SELECT("complex_multiplication_2d_SLM"));
  int argIdx = 0;
  kernel.arg(argIdx++, WrapHandle((cl_mem)dst, &ctx));
  kernel.arg(argIdx++, offset_offset_dst << 1);
  kernel.arg(argIdx++, local_mem_size_in_bytes);
  kernel.arg(argIdx++, WrapHandle((cl_mem)src1, &ctx));
  kernel.arg(argIdx++, offset_offset_src1 << 1);
  kernel.arg(argIdx++, WrapHandle((cl_mem)src2, &ctx));
  kernel.arg(argIdx++, offset_offset_src2 << 1);
  kernel.arg(argIdx++, num_output);
  kernel.arg(argIdx++, map_size_in_dtype4);
  kernel.arg(argIdx++, ch_gr);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(), kernel, 2,
      NULL, global_work_size, local_work_size, 0, NULL, &event));
  kernel_execution_time(&event, "complex_multiplication_2d_SLM");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
      kernel.handle().get(), 2, NULL, global_work_size, local_work_size, 0,
      NULL, NULL));
#endif
}
template void caffe_gpu_elementMul_2D_SLM<float>(DtypeComplex<float>* dst,
    const DtypeComplex<float>* src1, const DtypeComplex<float>* src2,
    const int size, const int ch_gr, const int num_output);
template void caffe_gpu_elementMul_2D_SLM<double>(DtypeComplex<double>* dst,
    const DtypeComplex<double>* src1, const DtypeComplex<double>* src2,
    const int size, const int ch_gr, const int num_output);

template <typename Dtype>
void caffe_gpu_elementMul_3D(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int size, const int ch_gr, const int out_gr, const int num_output) {
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();

  int offset_offset_dst = 0;
  int offset_offset_src1 = 0;
  int offset_offset_src2 = 0;

  // Dim 1: # of Dtype2
  const size_t global_work_size[3] = { (size_t)size, (size_t)ch_gr,
                                       (size_t)num_output };
  viennacl::ocl::kernel kernel = ctx.get_kernel("kernel_program",
      CL_KERNEL_SELECT("complex_multiplication_3d"));
  int argIdx = 0;
  kernel.arg(argIdx++, WrapHandle((cl_mem)dst, &ctx));
  kernel.arg(argIdx++, offset_offset_dst << 1);
  kernel.arg(argIdx++, WrapHandle((cl_mem)src1, &ctx));
  kernel.arg(argIdx++, offset_offset_src1 << 1);
  kernel.arg(argIdx++, WrapHandle((cl_mem)src2, &ctx));
  kernel.arg(argIdx++, offset_offset_src2 << 1);
  kernel.arg(argIdx++, size);
  kernel.arg(argIdx++, ch_gr);
  kernel.arg(argIdx++, out_gr);
  kernel.arg(argIdx++, num_output);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(), kernel, 3,
      NULL, global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "complex_multiplication_3d");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
      kernel.handle().get(), 3, NULL, global_work_size, NULL, 0,
      NULL, NULL));
#endif
}
template void caffe_gpu_elementMul_3D<float>(DtypeComplex<float>* dst,
    const DtypeComplex<float>* src1, const DtypeComplex<float>* src2,
    const int size, const int ch_gr, const int out_gr, const int num_output);
template void caffe_gpu_elementMul_3D<double>(DtypeComplex<double>* dst,
    const DtypeComplex<double>* src1, const DtypeComplex<double>* src2,
    const int size, const int ch_gr, const int out_gr, const int num_output);

template <typename Dtype>
void caffe_gpu_fft_execute_r2c(clfftPlanHandle plan, const Dtype* in,
    DtypeComplex<Dtype>* out) {
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();
  cl_command_queue queue = ctx.get_queue().handle().get();

#ifdef DEBUG_PROFILE
  cl_event event = 0;
  CLFFT_CHECK(clfftEnqueueTransform(plan, CLFFT_FORWARD, 1, &queue,
      0, NULL, &event, &mem_in, &mem_out, NULL));
  kernel_execution_time(&event, "clfft R2C");
#else
  CLFFT_CHECK(clfftEnqueueTransform(plan, CLFFT_FORWARD, 1, &queue,
      0, NULL, NULL,
      reinterpret_cast<cl_mem*>(reinterpret_cast<uintptr_t>(&in)),
      reinterpret_cast<cl_mem*>(reinterpret_cast<uintptr_t>(&out)),
      NULL));
#endif
}
template void caffe_gpu_fft_execute_r2c<float>(clfftPlanHandle plan,
    const float* in, DtypeComplex<float>* out);
template void caffe_gpu_fft_execute_r2c<double>(clfftPlanHandle plan,
    const double* in, DtypeComplex<double>* out);

template <typename Dtype>
void caffe_gpu_fft_execute_c2r(clfftPlanHandle plan,
    const DtypeComplex<Dtype>* in, Dtype* out) {
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();
  cl_command_queue queue = ctx.get_queue().handle().get();

#ifdef DEBUG_PROFILE
  cl_event event = 0;
  CLFFT_CHECK(clfftEnqueueTransform(plan, CLFFT_BACKWARD, 1, &queue,
      0, NULL, &event, &mem_in, &mem_out, NULL));
  kernel_execution_time(&event, "clfft C2R");
#else
  CLFFT_CHECK(clfftEnqueueTransform(plan, CLFFT_BACKWARD, 1, &queue,
      0, NULL, NULL,
      reinterpret_cast<cl_mem*>(reinterpret_cast<uintptr_t>(&in)),
      reinterpret_cast<cl_mem*>(reinterpret_cast<uintptr_t>(&out)),
      NULL));
#endif
}
template void caffe_gpu_fft_execute_c2r<float>(clfftPlanHandle plan,
    const DtypeComplex<float>* in, float* out);
template void caffe_gpu_fft_execute_c2r<double>(clfftPlanHandle plan,
    const DtypeComplex<double>* in, double* out);

template <typename Dtype>
void caffe_gpu_fft_execute_r2c_inplace(clfftPlanHandle plan, Dtype* inout) {
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();
  cl_command_queue queue = ctx.get_queue().handle().get();

#ifdef DEBUG_PROFILE
  cl_event event = 0;
  CLFFT_CHECK(clfftEnqueueTransform(plan, CLFFT_FORWARD, 1, &queue,
      0, NULL, &event, &mem_inout, NULL, NULL));
  kernel_execution_time(&event, "clfft In-place R2C");
#else
  CLFFT_CHECK(clfftEnqueueTransform(plan, CLFFT_FORWARD, 1, &queue,
      0, NULL, NULL,
      reinterpret_cast<cl_mem*>(reinterpret_cast<uintptr_t>(&inout)),
      NULL, NULL));
#endif
}
template void caffe_gpu_fft_execute_r2c_inplace<float>(
    clfftPlanHandle plan, float* inout);
template void caffe_gpu_fft_execute_r2c_inplace<double>(
    clfftPlanHandle plan, double* inout);

template <typename Dtype>
void reshape_weights(DtypeComplex<Dtype>* dst, DtypeComplex<Dtype>* src,
    const int size, const int num_output, const int ch_gr) {
  viennacl::ocl::context &ctx = viennacl::ocl::current_context();

  cl_command_queue queue = ctx.get_queue().handle().get();
  viennacl::ocl::kernel kernel = ctx.get_kernel("kernel_program",
      CL_KERNEL_SELECT("convert_weight_to_channel_major"));
  int argIdx = 0;
  size_t global_work_size[2] = { (size_t)size, (size_t)num_output };
  kernel.arg(argIdx++, WrapHandle((cl_mem)dst, &ctx));
  kernel.arg(argIdx++, WrapHandle((cl_mem)src, &ctx));
  kernel.arg(argIdx++, size);
  kernel.arg(argIdx++, ch_gr);
  kernel.arg(argIdx++, num_output);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
      global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "Reshape weight to channel major");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel.handle().get(), 2, NULL,
      global_work_size, NULL, 0, NULL, NULL));
#endif
}
template void reshape_weights<float>(DtypeComplex<float>* dst,
    DtypeComplex<float>* src,
    const int size, const int num_output, const int ch_gr);
template void reshape_weights<double>(DtypeComplex<double>* dst,
    DtypeComplex<double>* src,
    const int size, const int num_output, const int ch_gr);

}  // namespace caffe
#endif  // USE_GREENTEA && USE_FFT
#endif  // !CPU_ONLY
