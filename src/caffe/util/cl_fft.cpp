#ifdef USE_OCL
#ifdef USE_FFT
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/fft.hpp"

extern "C" const char _cl_fft_start;
extern "C" const char _cl_fft_end;

// #define DEBUG_PROFILE

namespace caffe {

void submit_program(ClState* state) {
  static const char * options = "-cl-mad-enable -cl-fast-relaxed-math";
  state->submit_program("fft", &_cl_fft_start, &_cl_fft_end, options);
}

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
  ClState& state = Caffe::cl_state();
  cl_command_queue queue = state.get_command_queue();
  ClMemOff<uint8_t> buf_data = state.get_buffer_mem(data);
  void* mapped_mem_ptr = clEnqueueMapBuffer(queue, buf_data.memobj, CL_TRUE,
      CL_MAP_WRITE, 0, size, 0, NULL, NULL, NULL);
  memset(mapped_mem_ptr, 0, size);
  clEnqueueUnmapMemObject(queue, buf_data.memobj, mapped_mem_ptr, 0, NULL,
      NULL);
}

template <typename Dtype>
void get_aligned_offset(size_t* aligned_offset, int* offset_offset,
    const Dtype* data) {
  ClState& state = Caffe::cl_state();
  ClMemOff<Dtype> buf_data = state.get_buffer_mem(data);
  uint mem_base_address_align =
      (state.get_properties().mem_base_addr_align / 8) / sizeof(Dtype);
  *aligned_offset = static_cast<int>(buf_data.offset /
      static_cast<float>(mem_base_address_align)) * mem_base_address_align;
  *offset_offset = ((buf_data.offset /
      static_cast<float>(mem_base_address_align)) - *aligned_offset /
      mem_base_address_align) * mem_base_address_align;
}
template void get_aligned_offset<float>(size_t* aligned_offset,
    int* offset_offset, const float* data);
template void get_aligned_offset<double>(size_t* aligned_offset,
    int* offset_offset, const double* data);

template <typename Dtype>
void get_aligned_offset(size_t* aligned_offset, int* offset_offset,
    const DtypeComplex<Dtype>* data) {
  ClState& state = Caffe::cl_state();
  ClMemOff< DtypeComplex<Dtype> > buf_data = state.get_buffer_mem(data);
  uint mem_base_address_align =
      (state.get_properties().mem_base_addr_align / 8) /
      sizeof(DtypeComplex<Dtype>);
  *aligned_offset = static_cast<int>(buf_data.offset /
      static_cast<float>(mem_base_address_align)) * mem_base_address_align;
  *offset_offset = ((buf_data.offset /
      static_cast<float>(mem_base_address_align)) - *aligned_offset /
      mem_base_address_align) * mem_base_address_align;
}
template void get_aligned_offset<float>(size_t* aligned_offset,
    int* offset_offset, const DtypeComplex<float>* data);
template void get_aligned_offset<double>(size_t* aligned_offset,
    int* offset_offset, const DtypeComplex<double>* data);

// Copy and cyclic-shift 0 padding of weights to FFT real buffer
template <typename Dtype>
void fft_gpu_copy2buffer(Dtype* fft_gpu_weights_real, const Dtype* weight,
    int num_output, int group, int channels, int ker_h, int ker_w,
    int ker_c_h, int ker_c_w, int fft_height, int fft_width) {
  ClState& state = Caffe::cl_state();
  submit_program(&state);

  size_t aligned_offset_fft_gpu_weights_real;
  int offset_offset_fft_gpu_weights_real;
  get_aligned_offset(&aligned_offset_fft_gpu_weights_real,
      &offset_offset_fft_gpu_weights_real, fft_gpu_weights_real);
  cl_mem mem_fft_gpu_weights_real = state.create_subbuffer(fft_gpu_weights_real,
      aligned_offset_fft_gpu_weights_real);

  size_t aligned_offset_weight;
  int offset_offset_weight;
  get_aligned_offset(&aligned_offset_weight, &offset_offset_weight, weight);
  cl_mem mem_weight = state.create_subbuffer(weight, aligned_offset_weight);

  const int ch_gr = channels / group;
  const int ker_size_ch_group = ker_h * ker_w * ch_gr;
  const size_t global_work_size = num_output * ker_size_ch_group;
  int argIdx = 0;
  const int ker_size = ker_h * ker_w;
  const int complex_width_len = 2*(fft_width/2 + 1);
  ClKernel kernel = state.get_kernel("copy2buffer_cyclic_shift_in");
  kernel.set_arg_mem(argIdx++, mem_fft_gpu_weights_real);
  kernel.set_arg(argIdx++, offset_offset_fft_gpu_weights_real);
  kernel.set_arg_mem(argIdx++, mem_weight);
  kernel.set_arg(argIdx++, offset_offset_weight);
  kernel.set_arg(argIdx++, ker_size);
  kernel.set_arg(argIdx++, ch_gr);
  kernel.set_arg(argIdx++, ker_size_ch_group);
  kernel.set_arg(argIdx++, ker_w);
  kernel.set_arg(argIdx++, ker_c_h);
  kernel.set_arg(argIdx++, ker_c_w);
  kernel.set_arg(argIdx++, fft_height);
  kernel.set_arg(argIdx++, fft_width);
  kernel.set_arg(argIdx++, complex_width_len);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 1, NULL,
      &global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "copy2buffer_cyclic_shift_in");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 1, NULL,
      &global_work_size, NULL, 0, NULL, NULL));
#endif
  clReleaseMemObject(mem_fft_gpu_weights_real);
  clReleaseMemObject(mem_weight);
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
/*
template <typename Dtype>
void fft_gpu_copy2buffer_in(Dtype* map_out, const Dtype* map_in,
    int height_out, int width_out, int height, int width,
    int stride_h, int stride_w, int pad_h, int pad_w) {
  ClState& state = Caffe::cl_state();
  submit_program(state);

  ClMemOff<Dtype> buf_map_out = state.get_buffer_mem(map_out);
  ClMemOff<Dtype> buf_map_in = state.get_buffer_mem(map_in);

  size_t aligned_offset_map_out;
  int offset_offset_map_out;
  get_aligned_offset(&aligned_offset_map_out, &offset_offset_map_out, map_out);
  cl_mem mem_map_out = state.create_subbuffer(map_out, aligned_offset_map_out);

  size_t aligned_offset_map_in;
  int offset_offset_map_in;
  get_aligned_offset(&aligned_offset_map_in, &offset_offset_map_in, map_in);
  cl_mem mem_map_in = state.create_subbuffer(map_in, aligned_offset_map_in);

  int size = height * width;
  const size_t global_work_size = size;
  ClKernel kernel;
  if (width < 4) {
    kernel = state.get_kernel("copy2buffer_left_top_in_naive");
  } else {
    kernel = state.get_kernel("copy2buffer_left_top_in");
  }
  int argIdx = 0;
  kernel.set_arg_mem(argIdx++, mem_map_out);
  kernel.set_arg(argIdx++, offset_offset_map_out);
  kernel.set_arg_mem(argIdx++, mem_map_in);
  kernel.set_arg(argIdx++, offset_offset_map_in);
  kernel.set_arg(argIdx++, size);
  kernel.set_arg(argIdx++, height_out);
  kernel.set_arg(argIdx++, width_out);
  kernel.set_arg(argIdx++, height);
  kernel.set_arg(argIdx++, width);
  kernel.set_arg(argIdx++, stride_h);
  kernel.set_arg(argIdx++, stride_w);
  kernel.set_arg(argIdx++, pad_h);
  kernel.set_arg(argIdx++, pad_w);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 1, NULL,
      &global_work_size, NULL, 0, NULL, &event));
  if (width < 4)
    kernel_execution_time(&event, "copy2buffer_left_top_in_naive");
  else
    kernel_execution_time(&event, "copy2buffer_left_top_in");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 1, NULL,
      &global_work_size, NULL, 0, NULL, NULL));
#endif

  clReleaseMemObject(mem_map_out);
  clReleaseMemObject(mem_map_in);
}
template void fft_gpu_copy2buffer_in<float>(float* map_out, const float* map_in,
    int height_out, int width_out, int height, int width,
    int stride_h, int stride_w, int pad_h, int pad_w);
template void fft_gpu_copy2buffer_in<double>(double* map_out,
    const double* map_in,
    int height_out, int width_out, int height, int width,
    int stride_h, int stride_w, int pad_h, int pad_w);
*/

// Copy and left-top 0 padding of data to FFT real buffer
template <typename Dtype>
void fft_gpu_copy2buffer_in_2D(Dtype* map_out, const Dtype* map_in,
    int channels, int height_out, int width_out, int height, int width,
    int stride_h, int stride_w, int pad_h, int pad_w) {
  ClState& state = Caffe::cl_state();
  submit_program(&state);

  size_t aligned_offset_map_out;
  int offset_offset_map_out;
  get_aligned_offset(&aligned_offset_map_out, &offset_offset_map_out, map_out);
  cl_mem mem_map_out = state.create_subbuffer(map_out, aligned_offset_map_out);

  size_t aligned_offset_map_in;
  int offset_offset_map_in;
  get_aligned_offset(&aligned_offset_map_in, &offset_offset_map_in, map_in);
  cl_mem mem_map_in = state.create_subbuffer(map_in, aligned_offset_map_in);

  int map_out_size = height_out * width_out;
  int size = height * width;
  int count = size >> 2;
  const size_t global_work_size[2] = { (size_t)size, (size_t)channels };
  ClKernel kernel;
  if (width < 4) {
    kernel = state.get_kernel("copy2buffer_left_top_in_naive_2d");
  } else {
    kernel = state.get_kernel("copy2buffer_left_top_in_2d");
  }
  int argIdx = 0;
  kernel.set_arg_mem(argIdx++, mem_map_out);
  kernel.set_arg(argIdx++, offset_offset_map_out);
  kernel.set_arg_mem(argIdx++, mem_map_in);
  kernel.set_arg(argIdx++, offset_offset_map_in);
  kernel.set_arg(argIdx++, map_out_size);
  kernel.set_arg(argIdx++, size);
  kernel.set_arg(argIdx++, count);
  kernel.set_arg(argIdx++, height_out);
  kernel.set_arg(argIdx++, width_out);
  kernel.set_arg(argIdx++, height);
  kernel.set_arg(argIdx++, width);
  kernel.set_arg(argIdx++, stride_h);
  kernel.set_arg(argIdx++, stride_w);
  kernel.set_arg(argIdx++, pad_h);
  kernel.set_arg(argIdx++, pad_w);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 2, NULL,
      global_work_size, NULL, 0, NULL, &event));
  if (width < 4)
    kernel_execution_time(&event, "copy2buffer_left_top_in_naive_2d");
  else
    kernel_execution_time(&event, "copy2buffer_left_top_in_2d");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 2, NULL,
      global_work_size, NULL, 0, NULL, NULL));
#endif

  clReleaseMemObject(mem_map_out);
  clReleaseMemObject(mem_map_in);
}
template void fft_gpu_copy2buffer_in_2D<float>(float* map_out,
    const float* map_in, int channels,
    int height_out, int width_out, int height, int width,
    int stride_h, int stride_w, int pad_h, int pad_w);
template void fft_gpu_copy2buffer_in_2D<double>(double* map_out,
    const double* map_in, int channels,
    int height_out, int width_out, int height, int width,
    int stride_h, int stride_w, int pad_h, int pad_w);

// Copy from left-top 0 padded data to real buffer
/*
template <typename Dtype>
void fft_gpu_copy2buffer_out_forward(Dtype* map_out, const Dtype* map_in,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
  ClState& state = Caffe::cl_state();
  submit_program(state);

  ClMemOff<Dtype> buf_map_out = state.get_buffer_mem(map_out);
  ClMemOff<Dtype> buf_map_in = state.get_buffer_mem(map_in);

  size_t aligned_offset_map_out;
  int offset_offset_map_out;
  get_aligned_offset(&aligned_offset_map_out, &offset_offset_map_out, map_out);
  cl_mem mem_map_out = state.create_subbuffer(map_out, aligned_offset_map_out);

  size_t aligned_offset_map_in;
  int offset_offset_map_in;
  get_aligned_offset(&aligned_offset_map_in, &offset_offset_map_in, map_in);
  cl_mem mem_map_in = state.create_subbuffer(map_in, aligned_offset_map_in);

  int size = height_out * width_out;
  const size_t global_work_size = size;
  ClKernel kernel;
  if (width_out < 4) {
    kernel = state.get_kernel("copy2buffer_left_top_out_naive");
  } else {
    kernel = state.get_kernel("copy2buffer_left_top_out");
  }
  int argIdx = 0;
  kernel.set_arg_mem(argIdx++, mem_map_out);
  kernel.set_arg(argIdx++, offset_offset_map_out);
  kernel.set_arg_mem(argIdx++, mem_map_in);
  kernel.set_arg(argIdx++, offset_offset_map_in);
  kernel.set_arg(argIdx++, size);
  kernel.set_arg(argIdx++, height_out);
  kernel.set_arg(argIdx++, width_out);
  kernel.set_arg(argIdx++, fft_height);
  kernel.set_arg(argIdx++, fft_width);
  kernel.set_arg(argIdx++, kernel_center_h);
  kernel.set_arg(argIdx++, kernel_center_w);
  kernel.set_arg(argIdx++, stride_h);
  kernel.set_arg(argIdx++, stride_w);
  kernel.set_arg(argIdx++, pad_h);
  kernel.set_arg(argIdx++, pad_w);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 1, NULL,
      &global_work_size, NULL, 0, NULL, &event));
  if (width_out < 4)
    kernel_execution_time(&event, "copy2buffer_left_top_out_naive");
  else
    kernel_execution_time(&event, "copy2buffer_left_top_out");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 1, NULL,
      &global_work_size, NULL, 0, NULL, NULL));
#endif

  clReleaseMemObject(mem_map_out);
  clReleaseMemObject(mem_map_in);
}
template void fft_gpu_copy2buffer_out_forward<float>(float* map_out,
    const float* map_in,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w);
template void fft_gpu_copy2buffer_out_forward<double>(double* map_out,
    const double* map_in,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w);
*/

// Copy from left-top 0 padded data to real buffer
template <typename Dtype>
void fft_gpu_copy2buffer_out_forward_2D(Dtype* map_out, const Dtype* map_in,
    int num_output,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
  ClState& state = Caffe::cl_state();
  submit_program(&state);

  size_t aligned_offset_map_out;
  int offset_offset_map_out;
  get_aligned_offset(&aligned_offset_map_out, &offset_offset_map_out, map_out);
  cl_mem mem_map_out = state.create_subbuffer(map_out, aligned_offset_map_out);

  size_t aligned_offset_map_in;
  int offset_offset_map_in;
  get_aligned_offset(&aligned_offset_map_in, &offset_offset_map_in, map_in);
  cl_mem mem_map_in = state.create_subbuffer(map_in, aligned_offset_map_in);

  int size = height_out * width_out;
  int count = size >> 2;
  int map_in_size = fft_height * fft_width;
  const size_t global_work_size[2] = { (size_t)size, (size_t)num_output };
  ClKernel kernel;
  if (width_out < 4) {
    kernel = state.get_kernel("copy2buffer_left_top_out_naive_2d");
  } else {
    kernel = state.get_kernel("copy2buffer_left_top_out_2d");
  }
  int argIdx = 0;
  kernel.set_arg_mem(argIdx++, mem_map_out);
  kernel.set_arg(argIdx++, offset_offset_map_out);
  kernel.set_arg_mem(argIdx++, mem_map_in);
  kernel.set_arg(argIdx++, offset_offset_map_in);
  kernel.set_arg(argIdx++, size);
  kernel.set_arg(argIdx++, count);
  kernel.set_arg(argIdx++, map_in_size);
  kernel.set_arg(argIdx++, height_out);
  kernel.set_arg(argIdx++, width_out);
  kernel.set_arg(argIdx++, fft_height);
  kernel.set_arg(argIdx++, fft_width);
  kernel.set_arg(argIdx++, kernel_center_h);
  kernel.set_arg(argIdx++, kernel_center_w);
  kernel.set_arg(argIdx++, stride_h);
  kernel.set_arg(argIdx++, stride_w);
  kernel.set_arg(argIdx++, pad_h);
  kernel.set_arg(argIdx++, pad_w);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 2, NULL,
      global_work_size, NULL, 0, NULL, &event));
  if (width_out < 4)
    kernel_execution_time(&event, "copy2buffer_left_top_out_naive_2d");
  else
    kernel_execution_time(&event, "copy2buffer_left_top_out_2d");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 2, NULL,
      global_work_size, NULL, 0, NULL, NULL));
#endif

  clReleaseMemObject(mem_map_out);
  clReleaseMemObject(mem_map_in);
}
template void fft_gpu_copy2buffer_out_forward_2D<float>(float* map_out,
    const float* map_in, int num_output,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w);
template void fft_gpu_copy2buffer_out_forward_2D<double>(double* map_out,
    const double* map_in, int num_output,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w);

template <typename Dtype>
void fft_gpu_copy2buffer_out_backward(Dtype* map_out, const Dtype* map_in,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
  ClState& state = Caffe::cl_state();
  submit_program(&state);

  size_t aligned_offset_map_out;
  int offset_offset_map_out;
  get_aligned_offset(&aligned_offset_map_out, &offset_offset_map_out, map_out);
  cl_mem mem_map_out = state.create_subbuffer(map_out, aligned_offset_map_out);

  size_t aligned_offset_map_in;
  int offset_offset_map_in;
  get_aligned_offset(&aligned_offset_map_in, &offset_offset_map_in, map_in);
  cl_mem mem_map_in = state.create_subbuffer(map_in, aligned_offset_map_in);

  const size_t global_work_size = height_out * width_out;
  ClKernel kernel = state.get_kernel("copy2buffer_cyclic_shift_out");
  int argIdx = 0;
  kernel.set_arg_mem(argIdx++, mem_map_out);
  kernel.set_arg(argIdx++, offset_offset_map_out);
  kernel.set_arg_mem(argIdx++, mem_map_in);
  kernel.set_arg(argIdx++, offset_offset_map_in);
  kernel.set_arg(argIdx++, width_out);
  kernel.set_arg(argIdx++, fft_height);
  kernel.set_arg(argIdx++, fft_width);
  kernel.set_arg(argIdx++, kernel_center_h);
  kernel.set_arg(argIdx++, kernel_center_w);
  kernel.set_arg(argIdx++, stride_h);
  kernel.set_arg(argIdx++, stride_w);
  kernel.set_arg(argIdx++, pad_h);
  kernel.set_arg(argIdx++, pad_w);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 1, NULL,
      &global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "copy2buffer_cyclic_shift_out");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 1, NULL,
      &global_work_size, NULL, 0, NULL, NULL));
#endif

  clReleaseMemObject(mem_map_out);
  clReleaseMemObject(mem_map_in);
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
void fft_gpu_copy2buffer_out_backward_2D(Dtype* map_out, const Dtype* map_in,
    int channels, int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
  ClState& state = Caffe::cl_state();
  submit_program(&state);

  size_t aligned_offset_map_out;
  int offset_offset_map_out;
  get_aligned_offset(&aligned_offset_map_out, &offset_offset_map_out, map_out);
  cl_mem mem_map_out = state.create_subbuffer(map_out, aligned_offset_map_out);

  size_t aligned_offset_map_in;
  int offset_offset_map_in;
  get_aligned_offset(&aligned_offset_map_in, &offset_offset_map_in, map_in);
  cl_mem mem_map_in = state.create_subbuffer(map_in, aligned_offset_map_in);

  int map_out_size = height_out * width_out;
  int map_in_size = fft_height * fft_width;
  const size_t global_work_size[2] = { (size_t)map_out_size, (size_t)channels };
  ClKernel kernel = state.get_kernel("copy2buffer_cyclic_shift_out_2d");
  int argIdx = 0;
  kernel.set_arg_mem(argIdx++, mem_map_out);
  kernel.set_arg(argIdx++, offset_offset_map_out);
  kernel.set_arg_mem(argIdx++, mem_map_in);
  kernel.set_arg(argIdx++, offset_offset_map_in);
  kernel.set_arg(argIdx++, map_out_size);
  kernel.set_arg(argIdx++, map_in_size);
  kernel.set_arg(argIdx++, width_out);
  kernel.set_arg(argIdx++, fft_height);
  kernel.set_arg(argIdx++, fft_width);
  kernel.set_arg(argIdx++, kernel_center_h);
  kernel.set_arg(argIdx++, kernel_center_w);
  kernel.set_arg(argIdx++, stride_h);
  kernel.set_arg(argIdx++, stride_w);
  kernel.set_arg(argIdx++, pad_h);
  kernel.set_arg(argIdx++, pad_w);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 2, NULL,
      global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "copy2buffer_cyclic_shift_out_2d");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 2, NULL,
      global_work_size, NULL, 0, NULL, NULL));
#endif

  clReleaseMemObject(mem_map_out);
  clReleaseMemObject(mem_map_in);
}
template void fft_gpu_copy2buffer_out_backward_2D<float>(float* map_out,
    const float* map_in, int channels,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w);
template void fft_gpu_copy2buffer_out_backward_2D<double>(double* map_out,
    const double* map_in, int channels,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w);

template <typename Dtype>
void caffe_gpu_elementMulConj_1D(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int map_size, const int ch_gr) {
  // Note: map_size is the number of DtypeComplex values
  ClState& state = Caffe::cl_state();
  submit_program(&state);

  size_t aligned_offset_dst;
  int offset_offset_dst;
  get_aligned_offset(&aligned_offset_dst, &offset_offset_dst, dst);
  cl_mem mem_dst = state.create_subbuffer(dst, aligned_offset_dst);

  size_t aligned_offset_src1;
  int offset_offset_src1;
  get_aligned_offset(&aligned_offset_src1, &offset_offset_src1, src1);
  cl_mem mem_src1 = state.create_subbuffer(src1, aligned_offset_src1);

  size_t aligned_offset_src2;
  int offset_offset_src2;
  get_aligned_offset(&aligned_offset_src2, &offset_offset_src2, src2);
  cl_mem mem_src2 = state.create_subbuffer(src2, aligned_offset_src2);

  const size_t global_work_size = map_size >> 1;
  ClKernel kernel = state.get_kernel("complex_conjugate_multiplication_1d");
  int argIdx = 0;
  kernel.set_arg_mem(argIdx++, mem_dst);
  kernel.set_arg(argIdx++, offset_offset_dst << 1);
  kernel.set_arg_mem(argIdx++, mem_src1);
  kernel.set_arg(argIdx++, offset_offset_src1 << 1);
  kernel.set_arg_mem(argIdx++, mem_src2);
  kernel.set_arg(argIdx++, offset_offset_src2 << 1);
  kernel.set_arg(argIdx++, ch_gr);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 1, NULL,
      &global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "complex_conjugate_multiplication_1d");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 1, NULL,
      &global_work_size, NULL, 0, NULL, NULL));
#endif

  clReleaseMemObject(mem_dst);
  clReleaseMemObject(mem_src1);
  clReleaseMemObject(mem_src2);
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
  ClState& state = Caffe::cl_state();
  submit_program(&state);

  size_t aligned_offset_dst;
  int offset_offset_dst;
  get_aligned_offset(&aligned_offset_dst, &offset_offset_dst, dst);
  cl_mem mem_dst = state.create_subbuffer(dst, aligned_offset_dst);

  size_t aligned_offset_src1;
  int offset_offset_src1;
  get_aligned_offset(&aligned_offset_src1, &offset_offset_src1, src1);
  cl_mem mem_src1 = state.create_subbuffer(src1, aligned_offset_src1);

  size_t aligned_offset_src2;
  int offset_offset_src2;
  get_aligned_offset(&aligned_offset_src2, &offset_offset_src2, src2);
  cl_mem mem_src2 = state.create_subbuffer(src2, aligned_offset_src2);

  cl_command_queue queue = state.get_command_queue();
  size_t block_size = map_size * ch_gr * sizeof(DtypeComplex<Dtype>);
  cl_mem src1_vec = clCreateBuffer(state.get_context(), CL_MEM_READ_WRITE,
      block_size, NULL, NULL);
  size_t global_work_size1 = map_size;
  ClKernel kernel = state.get_kernel("convert_data_to_channel_major");
  int argIdx = 0;
  kernel.set_arg_mem(argIdx++, src1_vec);
  kernel.set_arg_mem(argIdx++, mem_src1);
  kernel.set_arg(argIdx++, map_size);
  kernel.set_arg(argIdx++, ch_gr);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
      &global_work_size1, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "Reshape data to channel major");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
      &global_work_size1, NULL, 0, NULL, NULL));
#endif

  ClKernel kernel_batchedCdotc = state.get_kernel("batchedCdotc");
  // Batched complex number dot product
  size_t global_work_size2[2] = { (size_t)map_size, (size_t)out_gr };
  argIdx = 0;
  kernel_batchedCdotc.set_arg_mem(argIdx++, mem_dst);
  kernel_batchedCdotc.set_arg_mem(argIdx++, src1_vec);
  kernel_batchedCdotc.set_arg_mem(argIdx++, mem_src2);
  kernel_batchedCdotc.set_arg(argIdx++, map_size);
  kernel_batchedCdotc.set_arg(argIdx++, ch_gr);
  kernel_batchedCdotc.set_arg(argIdx++, out_gr);
#ifdef DEBUG_PROFILE
  OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel_batchedCdotc, 2, NULL,
      global_work_size2, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "Batched complex dot product");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel_batchedCdotc, 2, NULL,
      global_work_size2, NULL, 0, NULL, NULL));
#endif
  clReleaseMemObject(src1_vec);
  clReleaseMemObject(mem_dst);
  clReleaseMemObject(mem_src1);
  clReleaseMemObject(mem_src2);
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
void caffe_gpu_elementMulConj_2D(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int out_gr, const int map_size, const int ch_gr) {
  // Note: map_size is the number of DtypeComplex values
  ClState& state = Caffe::cl_state();
  submit_program(&state);

  size_t aligned_offset_dst;
  int offset_offset_dst;
  get_aligned_offset(&aligned_offset_dst, &offset_offset_dst, dst);
  cl_mem mem_dst = state.create_subbuffer(dst, aligned_offset_dst);

  size_t aligned_offset_src1;
  int offset_offset_src1;
  get_aligned_offset(&aligned_offset_src1, &offset_offset_src1, src1);
  cl_mem mem_src1 = state.create_subbuffer(src1, aligned_offset_src1);

  size_t aligned_offset_src2;
  int offset_offset_src2;
  get_aligned_offset(&aligned_offset_src2, &offset_offset_src2, src2);
  cl_mem mem_src2 = state.create_subbuffer(src2, aligned_offset_src2);

  const size_t global_work_size[2] = { (size_t)map_size >> 1, (size_t)out_gr };
  ClKernel kernel = state.get_kernel("complex_conjugate_multiplication_2d");
  int argIdx = 0;
  kernel.set_arg_mem(argIdx++, mem_dst);
  kernel.set_arg(argIdx++, offset_offset_dst << 1);
  kernel.set_arg_mem(argIdx++, mem_src1);
  kernel.set_arg(argIdx++, offset_offset_src1 << 1);
  kernel.set_arg_mem(argIdx++, mem_src2);
  kernel.set_arg(argIdx++, offset_offset_src2 << 1);
  kernel.set_arg(argIdx++, out_gr);
  kernel.set_arg(argIdx++, map_size >> 1);
  kernel.set_arg(argIdx++, ch_gr);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 2, NULL,
      global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "complex_conjugate_multiplication_2d");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 2, NULL,
      global_work_size, NULL, 0, NULL, NULL));
#endif

  clReleaseMemObject(mem_dst);
  clReleaseMemObject(mem_src1);
  clReleaseMemObject(mem_src2);
}
template void caffe_gpu_elementMulConj_2D<float>(DtypeComplex<float>* dst,
    const DtypeComplex<float>* src1, const DtypeComplex<float>* src2,
    const int out_gr, const int map_size, const int ch_gr);
template void caffe_gpu_elementMulConj_2D<double>(DtypeComplex<double>* dst,
    const DtypeComplex<double>* src1, const DtypeComplex<double>* src2,
    const int out_gr, const int map_size, const int ch_gr);

template <typename Dtype>
void caffe_gpu_elementMulConj_2D_SLM(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int out_gr, const int map_size, const int ch_gr) {
  // Note: size is the number of DtypeComplex values
  ClState& state = Caffe::cl_state();
  submit_program(&state);

  size_t aligned_offset_dst;
  int offset_offset_dst;
  get_aligned_offset(&aligned_offset_dst, &offset_offset_dst, dst);
  cl_mem mem_dst = state.create_subbuffer(dst, aligned_offset_dst);

  size_t aligned_offset_src1;
  int offset_offset_src1;
  get_aligned_offset(&aligned_offset_src1, &offset_offset_src1, src1);
  cl_mem mem_src1 = state.create_subbuffer(src1, aligned_offset_src1);

  size_t aligned_offset_src2;
  int offset_offset_src2;
  get_aligned_offset(&aligned_offset_src2, &offset_offset_src2, src2);
  cl_mem mem_src2 = state.create_subbuffer(src2, aligned_offset_src2);

  int map_float4_size = map_size >> 1;
  // Note:
  // (16, 1) is good for Unit Test
  // (32, 16) is good for CaffNet
  // (128, 4) is perf hint recommended
  int local_work_size_x = (map_float4_size < 512) ? 16 : 32;  // TODO: Temporary
  int local_work_size_y = (out_gr < 16) ? 1 : 16;  // TODO: Temporary
  if (out_gr >=  16 &&
      state.get_properties().device_max_work_group_size < 512) {
    local_work_size_y = 8;
  }
  const size_t local_work_size[2] = { (size_t)local_work_size_x, (size_t)local_work_size_y };
  int global_work_size_x =
      CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(map_float4_size, local_work_size_x);
  int global_work_size_y =
      CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(out_gr, local_work_size_y);
  const size_t global_work_size[2] = { (size_t)global_work_size_x, (size_t)global_work_size_y };
  ClKernel kernel = state.get_kernel(
      "complex_conjugate_multiplication_2d_SLM");
  int argIdx = 0;
  kernel.set_arg_mem(argIdx++, mem_dst);
  kernel.set_arg(argIdx++, offset_offset_dst << 1);
  kernel.set_arg_mem(argIdx++, mem_src1);
  kernel.set_arg(argIdx++, offset_offset_src1 << 1);
  kernel.set_arg_mem_local(
      argIdx++, ch_gr * local_work_size_x * sizeof(Dtype) * 4);
  kernel.set_arg_mem(argIdx++, mem_src2);
  kernel.set_arg(argIdx++, offset_offset_src2 << 1);
  kernel.set_arg(argIdx++, out_gr);
  kernel.set_arg(argIdx++, map_float4_size);
  kernel.set_arg(argIdx++, ch_gr);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 2, NULL,
      global_work_size, local_work_size, 0, NULL, &event));
  kernel_execution_time(&event, "complex_conjugate_multiplication_2d_SLM");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 2, NULL,
      global_work_size, local_work_size, 0, NULL, NULL));
#endif

  clReleaseMemObject(mem_dst);
  clReleaseMemObject(mem_src1);
  clReleaseMemObject(mem_src2);
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
  ClState& state = Caffe::cl_state();
  submit_program(&state);

  size_t aligned_offset_dst;
  int offset_offset_dst;
  get_aligned_offset(&aligned_offset_dst, &offset_offset_dst, dst);
  cl_mem mem_dst = state.create_subbuffer(dst, aligned_offset_dst);

  size_t aligned_offset_src1;
  int offset_offset_src1;
  get_aligned_offset(&aligned_offset_src1, &offset_offset_src1, src1);
  cl_mem mem_src1 = state.create_subbuffer(src1, aligned_offset_src1);

  size_t aligned_offset_src2;
  int offset_offset_src2;
  get_aligned_offset(&aligned_offset_src2, &offset_offset_src2, src2);
  cl_mem mem_src2 = state.create_subbuffer(src2, aligned_offset_src2);

  const size_t global_work_size[3] = { (size_t)map_size >> 1, (size_t)out_gr, (size_t)ch_gr };
  ClKernel kernel = state.get_kernel("complex_conjugate_multiplication_3d");
  int argIdx = 0;
  kernel.set_arg_mem(argIdx++, mem_dst);
  kernel.set_arg(argIdx++, offset_offset_dst << 1);
  kernel.set_arg_mem(argIdx++, mem_src1);
  kernel.set_arg(argIdx++, offset_offset_src1 << 1);
  kernel.set_arg_mem(argIdx++, mem_src2);
  kernel.set_arg(argIdx++, offset_offset_src2 << 1);
  kernel.set_arg(argIdx++, out_gr);
  kernel.set_arg(argIdx++, map_size >> 1);
  kernel.set_arg(argIdx++, ch_gr);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 3, NULL,
      global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "complex_conjugate_multiplication_3d");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 3, NULL,
      global_work_size, NULL, 0, NULL, NULL));
#endif

  clReleaseMemObject(mem_dst);
  clReleaseMemObject(mem_src1);
  clReleaseMemObject(mem_src2);
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
  ClState& state = Caffe::cl_state();
  submit_program(&state);

  size_t aligned_offset_dst;
  int offset_offset_dst;
  get_aligned_offset(&aligned_offset_dst, &offset_offset_dst, dst);
  cl_mem mem_dst = state.create_subbuffer(dst, aligned_offset_dst);

  size_t aligned_offset_src1;
  int offset_offset_src1;
  get_aligned_offset(&aligned_offset_src1, &offset_offset_src1, src1);
  cl_mem mem_src1 = state.create_subbuffer(src1, aligned_offset_src1);

  size_t aligned_offset_src2;
  int offset_offset_src2;
  get_aligned_offset(&aligned_offset_src2, &offset_offset_src2, src2);
  cl_mem mem_src2 = state.create_subbuffer(src2, aligned_offset_src2);

  int map_float4_size = map_size >> 1;
  // Note:
  // (16, 1) is good for Unit Test
  // (32, 2) is good for CaffNet
  // (128, 4) is perf hint recommended
  int local_work_size_x = (map_float4_size < 512) ? 16 : 32;  // TODO: Temporary
  int local_work_size_y = (out_gr < 16) ? 1 : 2;  // TODO: Temporary
  int local_work_size_z = 1;
  const size_t local_work_size[3] = {
      (size_t)local_work_size_x, (size_t)local_work_size_y, (size_t)local_work_size_z };
  int global_work_size_x =
      CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(map_float4_size, local_work_size_x);
  int global_work_size_y =
      CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(out_gr, local_work_size_y);
  int global_work_size_z =
      CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(ch_gr, local_work_size_z);
  const size_t global_work_size[3] = {
      (size_t)global_work_size_x, (size_t)global_work_size_y, (size_t)global_work_size_z };
  ClKernel kernel = state.get_kernel(
      "complex_conjugate_multiplication_3d_SLM");
  int argIdx = 0;
  kernel.set_arg_mem(argIdx++, mem_dst);
  kernel.set_arg(argIdx++, offset_offset_dst << 1);
  kernel.set_arg_mem_local(
      argIdx++, ch_gr * sizeof(Dtype) * 4);
  kernel.set_arg_mem(argIdx++, mem_src1);
  kernel.set_arg(argIdx++, offset_offset_src1 << 1);
  kernel.set_arg_mem_local(
      argIdx++, ch_gr * local_work_size_x * sizeof(Dtype) * 4);
  kernel.set_arg_mem(argIdx++, mem_src2);
  kernel.set_arg(argIdx++, offset_offset_src2 << 1);
  kernel.set_arg(argIdx++, out_gr);
  kernel.set_arg(argIdx++, map_float4_size);
  kernel.set_arg(argIdx++, ch_gr);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 3, NULL,
      global_work_size, local_work_size, 0, NULL, &event));
  kernel_execution_time(&event, "complex_conjugate_multiplication_3d_SLM");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 3, NULL,
      global_work_size, local_work_size, 0, NULL, NULL));
#endif

  clReleaseMemObject(mem_dst);
  clReleaseMemObject(mem_src1);
  clReleaseMemObject(mem_src2);
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
  ClState& state = Caffe::cl_state();
  submit_program(&state);

  size_t aligned_offset_dst;
  int offset_offset_dst;
  get_aligned_offset(&aligned_offset_dst, &offset_offset_dst, dst);
  cl_mem mem_dst = state.create_subbuffer(dst, aligned_offset_dst);

  size_t aligned_offset_src1;
  int offset_offset_src1;
  get_aligned_offset(&aligned_offset_src1, &offset_offset_src1, src1);
  cl_mem mem_src1 = state.create_subbuffer(src1, aligned_offset_src1);

  size_t aligned_offset_src2;
  int offset_offset_src2;
  get_aligned_offset(&aligned_offset_src2, &offset_offset_src2, src2);
  cl_mem mem_src2 = state.create_subbuffer(src2, aligned_offset_src2);

  const size_t global_work_size = size >> 1;  // # of Dtype4
  ClKernel kernel = state.get_kernel("complex_multiplication_1d");
  int argIdx = 0;
  kernel.set_arg_mem(argIdx++, mem_dst);
  kernel.set_arg(argIdx++, offset_offset_dst << 1);
  kernel.set_arg_mem(argIdx++, mem_src1);
  kernel.set_arg(argIdx++, offset_offset_src1 << 1);
  kernel.set_arg_mem(argIdx++, mem_src2);
  kernel.set_arg(argIdx++, offset_offset_src2 << 1);
  kernel.set_arg(argIdx++, size >> 1);
  kernel.set_arg(argIdx++, ch_gr);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 1, NULL,
      &global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "complex_multiplication_1d");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 1, NULL,
      &global_work_size, NULL, 0, NULL, NULL));
#endif

  clReleaseMemObject(mem_dst);
  clReleaseMemObject(mem_src1);
  clReleaseMemObject(mem_src2);
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
  ClState& state = Caffe::cl_state();
  submit_program(&state);

  size_t aligned_offset_dst;
  int offset_offset_dst;
  get_aligned_offset(&aligned_offset_dst, &offset_offset_dst, dst);
  cl_mem mem_dst = state.create_subbuffer(dst, aligned_offset_dst);

  size_t aligned_offset_src1;
  int offset_offset_src1;
  get_aligned_offset(&aligned_offset_src1, &offset_offset_src1, src1);
  cl_mem mem_src1 = state.create_subbuffer(src1, aligned_offset_src1);

  size_t aligned_offset_src2;
  int offset_offset_src2;
  get_aligned_offset(&aligned_offset_src2, &offset_offset_src2, src2);
  cl_mem mem_src2 = state.create_subbuffer(src2, aligned_offset_src2);

  // (16,2)=6K, (8,4)=1.5K work for CaffeNet
  // (128, 4) is perf hint recommended
  int local_work_size_x = 16;  // TODO: what is the best number?
  int local_work_size_y = 2;   // TODO: what is the best number?
  const size_t local_work_size[2] = { (size_t)local_work_size_x, (size_t)local_work_size_y };

  int map_size_in_dtype4 = size >> 1;  // # of Dtype4
  int global_work_size_x =
      CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(map_size_in_dtype4, local_work_size_x);
  int global_work_size_y =
      CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(num_output, local_work_size_y);
  const size_t global_work_size[2] = { (size_t)global_work_size_x, (size_t)global_work_size_y };
  const size_t local_mem_size_in_bytes =
      ch_gr * local_work_size_x * local_work_size_y * sizeof(Dtype) * 4;

  ClKernel kernel = state.get_kernel("complex_multiplication_2d_SLM");
  int argIdx = 0;
  kernel.set_arg_mem(argIdx++, mem_dst);
  kernel.set_arg(argIdx++, offset_offset_dst << 1);
  kernel.set_arg_mem_local(argIdx++, local_mem_size_in_bytes);
  kernel.set_arg_mem(argIdx++, mem_src1);
  kernel.set_arg(argIdx++, offset_offset_src1 << 1);
  kernel.set_arg_mem(argIdx++, mem_src2);
  kernel.set_arg(argIdx++, offset_offset_src2 << 1);
  kernel.set_arg(argIdx++, num_output);
  kernel.set_arg(argIdx++, map_size_in_dtype4);
  kernel.set_arg(argIdx++, ch_gr);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 2, NULL,
      global_work_size, local_work_size, 0, NULL, &event));
  kernel_execution_time(&event, "complex_multiplication_2d_SLM");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 2, NULL,
      global_work_size, local_work_size, 0, NULL, NULL));
#endif

  clReleaseMemObject(mem_dst);
  clReleaseMemObject(mem_src1);
  clReleaseMemObject(mem_src2);
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
  ClState& state = Caffe::cl_state();
  submit_program(&state);

  size_t aligned_offset_dst;
  int offset_offset_dst;
  get_aligned_offset(&aligned_offset_dst, &offset_offset_dst, dst);
  cl_mem mem_dst = state.create_subbuffer(dst, aligned_offset_dst);

  size_t aligned_offset_src1;
  int offset_offset_src1;
  get_aligned_offset(&aligned_offset_src1, &offset_offset_src1, src1);
  cl_mem mem_src1 = state.create_subbuffer(src1, aligned_offset_src1);

  size_t aligned_offset_src2;
  int offset_offset_src2;
  get_aligned_offset(&aligned_offset_src2, &offset_offset_src2, src2);
  cl_mem mem_src2 = state.create_subbuffer(src2, aligned_offset_src2);

  // Dim 1: # of Dtype2
  const size_t global_work_size[3] = { (size_t)size, (size_t)ch_gr, (size_t)num_output };
  ClKernel kernel = state.get_kernel("complex_multiplication_3d");
  int argIdx = 0;
  kernel.set_arg_mem(argIdx++, mem_dst);
  kernel.set_arg(argIdx++, offset_offset_dst << 1);
  kernel.set_arg_mem(argIdx++, mem_src1);
  kernel.set_arg(argIdx++, offset_offset_src1 << 1);
  kernel.set_arg_mem(argIdx++, mem_src2);
  kernel.set_arg(argIdx++, offset_offset_src2 << 1);
  kernel.set_arg(argIdx++, size);
  kernel.set_arg(argIdx++, ch_gr);
  kernel.set_arg(argIdx++, out_gr);
  kernel.set_arg(argIdx++, num_output);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 3, NULL,
      global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "complex_multiplication_3d");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 3, NULL,
      global_work_size, NULL, 0, NULL, NULL));
#endif

  clReleaseMemObject(mem_dst);
  clReleaseMemObject(mem_src1);
  clReleaseMemObject(mem_src2);
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
  ClState& state = Caffe::cl_state();
  cl_command_queue queue = state.get_command_queue();

  ClMemOff<Dtype> buf_in = state.get_buffer_mem(in);
  ClMemOff< DtypeComplex<Dtype> > buf_out = state.get_buffer_mem(out);

  cl_mem mem_in = state.create_subbuffer(in, buf_in.offset);
  cl_mem mem_out = state.create_subbuffer(out, buf_out.offset);

#ifdef DEBUG_PROFILE
  cl_event event = 0;
  CLFFT_CHECK(clfftEnqueueTransform(plan, CLFFT_FORWARD, 1, &queue,
      0, NULL, &event, &mem_in, &mem_out, NULL));
  kernel_execution_time(&event, "clfft R2C");
#else
  CLFFT_CHECK(clfftEnqueueTransform(plan, CLFFT_FORWARD, 1, &queue,
      0, NULL, NULL, &mem_in, &mem_out, NULL));
#endif

  clReleaseMemObject(mem_in);
  clReleaseMemObject(mem_out);
}
template void caffe_gpu_fft_execute_r2c<float>(clfftPlanHandle plan,
    const float* in, DtypeComplex<float>* out);
template void caffe_gpu_fft_execute_r2c<double>(clfftPlanHandle plan,
    const double* in, DtypeComplex<double>* out);

template <typename Dtype>
void caffe_gpu_fft_execute_c2r(clfftPlanHandle plan,
    const DtypeComplex<Dtype>* in, Dtype* out) {
  ClState& state = Caffe::cl_state();
  cl_command_queue queue = state.get_command_queue();

  ClMemOff< DtypeComplex<Dtype> > buf_in = state.get_buffer_mem(in);
  ClMemOff<Dtype> buf_out = state.get_buffer_mem(out);

  cl_mem mem_in = state.create_subbuffer(in, buf_in.offset);
  cl_mem mem_out = state.create_subbuffer(out, buf_out.offset);

#ifdef DEBUG_PROFILE
  cl_event event = 0;
  CLFFT_CHECK(clfftEnqueueTransform(plan, CLFFT_BACKWARD, 1, &queue,
      0, NULL, &event, &mem_in, &mem_out, NULL));
  kernel_execution_time(&event, "clfft C2R");
#else
  CLFFT_CHECK(clfftEnqueueTransform(plan, CLFFT_BACKWARD, 1, &queue,
      0, NULL, NULL, &mem_in, &mem_out, NULL));
#endif

  clReleaseMemObject(mem_in);
  clReleaseMemObject(mem_out);
}
template void caffe_gpu_fft_execute_c2r<float>(clfftPlanHandle plan,
    const DtypeComplex<float>* in, float* out);
template void caffe_gpu_fft_execute_c2r<double>(clfftPlanHandle plan,
    const DtypeComplex<double>* in, double* out);

template <typename Dtype>
void caffe_gpu_fft_execute_r2c_inplace(clfftPlanHandle plan, Dtype* inout) {
  ClState& state = Caffe::cl_state();
  cl_command_queue queue = state.get_command_queue();

  ClMemOff<Dtype> buf_inout = state.get_buffer_mem(inout);
  cl_mem mem_inout = state.create_subbuffer(inout, buf_inout.offset);

#ifdef DEBUG_PROFILE
  cl_event event = 0;
  CLFFT_CHECK(clfftEnqueueTransform(plan, CLFFT_FORWARD, 1, &queue,
      0, NULL, &event, &mem_inout, NULL, NULL));
  kernel_execution_time(&event, "clfft In-place R2C");
#else
  CLFFT_CHECK(clfftEnqueueTransform(plan, CLFFT_FORWARD, 1, &queue,
      0, NULL, NULL, &mem_inout, NULL, NULL));
#endif

  clReleaseMemObject(mem_inout);
}
template void caffe_gpu_fft_execute_r2c_inplace<float>(
    clfftPlanHandle plan, float* inout);
template void caffe_gpu_fft_execute_r2c_inplace<double>(
    clfftPlanHandle plan, double* inout);

template <typename Dtype>
void reshape_weights(DtypeComplex<Dtype>* dst, DtypeComplex<Dtype>* src,
    const int size, const int num_output, const int ch_gr) {
  ClState& state = Caffe::cl_state();
  submit_program(&state);

  size_t aligned_offset_dst;
  int offset_offset_dst;
  get_aligned_offset(&aligned_offset_dst, &offset_offset_dst, dst);
  cl_mem mem_dst = state.create_subbuffer(dst, aligned_offset_dst);

  size_t aligned_offset_src;
  int offset_offset_src;
  get_aligned_offset(&aligned_offset_src, &offset_offset_src, src);
  cl_mem mem_src = state.create_subbuffer(src, aligned_offset_src);

  cl_command_queue queue = state.get_command_queue();
  ClKernel kernel = state.get_kernel("convert_weight_to_channel_major");
  int argIdx = 0;
  size_t global_work_size[2] = { (size_t)size, (size_t)num_output };
  kernel.set_arg_mem(argIdx++, mem_dst);
  kernel.set_arg_mem(argIdx++, mem_src);
  kernel.set_arg(argIdx++, size);
  kernel.set_arg(argIdx++, ch_gr);
  kernel.set_arg(argIdx++, num_output);
#ifdef DEBUG_PROFILE
  cl_event event = 0;
  OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
      global_work_size, NULL, 0, NULL, &event));
  kernel_execution_time(&event, "Reshape weight to channel major");
#else
  OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
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
#endif  // USE_FFT
#endif  // USE_OCL
