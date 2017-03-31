/*
 * greentea_math_functions.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: Fabian Tschopp
 */

#include "caffe/common.hpp"
#include "caffe/device.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"

#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <boost/thread.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"

#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"

#if defined(USE_CLBLAS)
  #include <clBLAS.h>       // NOLINT
#elif defined(USE_CLBLAST)
  #include <clblast.h>      // NOLINT
#endif
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"


// ViennaCL 1.5.1 compability fix
#ifndef VIENNACL_MINOR_VERSION
#define VIENNACL_MINOR_VERSION 5
#endif

#if VIENNACL_MINOR_VERSION > 5
#define VCL_ROW_MAJOR , true
#define VCL_COL_MAJOR , false
#else
#define VCL_ROW_MAJOR
#define VCL_COL_MAJOR
#endif

namespace caffe {

void greentea_memset(const int_tp ctx_id, const uint_tp N, const int_tp alpha,
                     cl_mem X, const int_tp offX) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  // OpenCL Version >= 1.2 approach
  // clEnqueueFillBuffer(ctx.get_queue().handle().get(),
  //  X, &alpha, sizeof(int_tp),
  //                     offX, N, 0, NULL, NULL);
  // OpenCL Version < 1.2 fallback
  typedef float Dtype;
  viennacl::ocl::kernel &oclk_fill = program.get_kernel(
      CL_KERNEL_SELECT("fillbuffer"));
  viennacl::ocl::enqueue(
      oclk_fill(static_cast<int_tp>(N), static_cast<unsigned char>(alpha),
                WrapHandle(X, &ctx), offX),
      ctx.get_queue());
}

// Copy from OpenCL buffer to main memory
void greentea_gpu_memcpy(const uint_tp N, const cl_mem X, const int_tp offX,
                         void *Y, viennacl::ocl::context *ctx) {
  if (Y != NULL) {
    clEnqueueReadBuffer(ctx->get_queue().handle().get(), X, CL_TRUE, offX, N, Y,
                        0,
                        NULL,
                        NULL);
  }
}

// Copy from main memory to OpenCL buffer
void greentea_gpu_memcpy(const uint_tp N, const void* X, cl_mem Y,
                         const int_tp offY, viennacl::ocl::context *ctx) {
  if (X != NULL) {
    clEnqueueWriteBuffer(ctx->get_queue().handle().get(), Y,
    CL_TRUE,
                         offY, N, X, 0, NULL, NULL);
  }
}

// Copy from OpenCL to OpenCL buffer
void greentea_gpu_memcpy(const uint_tp N, const cl_mem X, const int_tp offX,
                         cl_mem Y, const int_tp offY,
                         viennacl::ocl::context *ctx) {
  clEnqueueCopyBuffer(ctx->get_queue().handle().get(), X, Y, offX, offY, N, 0,
  NULL,
                      NULL);
}

template<typename Dtype>
void greentea_copy(const int_tp N, const cl_mem X, const int_tp offX, Dtype* Y,
                   viennacl::ocl::context *ctx) {
  greentea_gpu_memcpy(sizeof(Dtype) * N, X, offX * sizeof(Dtype), Y, ctx);
}

template<typename Dtype>
void greentea_copy(const int_tp N, const Dtype* X, cl_mem Y, const int_tp offY,
                   viennacl::ocl::context *ctx) {
  greentea_gpu_memcpy(sizeof(Dtype) * N, X, Y, offY * sizeof(Dtype), ctx);
}

// Copy from OpenCL buffer to OpenCL buffer
template<typename Dtype>
void greentea_copy(const int_tp N, const cl_mem X, const int_tp offX, cl_mem Y,
                   const int_tp offY, viennacl::ocl::context *ctx) {
  greentea_gpu_memcpy(sizeof(Dtype) * N, X, offX * sizeof(Dtype), Y,
                      offY * sizeof(Dtype), ctx);
}

// Explicit instantiations
template void greentea_copy<int_tp>(const int_tp N, const cl_mem X,
                                    const int_tp offX,
                                    int_tp* Y,
                                    viennacl::ocl::context *ctx);
template void greentea_copy<uint_tp>(const int_tp N, const cl_mem X,
                                     const int_tp offX, uint_tp* Y,
                                     viennacl::ocl::context *ctx);
template void greentea_copy<float>(const int_tp N, const cl_mem X,
                                   const int_tp offX, float* Y,
                                   viennacl::ocl::context *ctx);
template void greentea_copy<double>(const int_tp N, const cl_mem X,
                                    const int_tp offX, double* Y,
                                    viennacl::ocl::context *ctx);
template void greentea_copy<int_tp>(const int_tp N, const int_tp* X, cl_mem Y,
                                    const int_tp offY,
                                    viennacl::ocl::context *ctx);
template void greentea_copy<uint_tp>(const int_tp N, const uint_tp* X, cl_mem Y,
                                     const int_tp offY,
                                     viennacl::ocl::context *ctx);
template void greentea_copy<float>(const int_tp N, const float* X, cl_mem Y,
                                   const int_tp offY,
                                   viennacl::ocl::context *ctx);
template void greentea_copy<double>(const int_tp N, const double* X, cl_mem Y,
                                    const int_tp offY,
                                    viennacl::ocl::context *ctx);
template void greentea_copy<int_tp>(const int_tp N, const cl_mem X,
                                    const int_tp offX, cl_mem Y,
                                    const int_tp offY,
                                    viennacl::ocl::context *ctx);
template void greentea_copy<uint_tp>(const int_tp N, const cl_mem X,
                                     const int_tp offX, cl_mem Y,
                                     const int_tp offY,
                                     viennacl::ocl::context *ctx);
template void greentea_copy<float>(const int_tp N, const cl_mem X,
                                   const int_tp offX, cl_mem Y,
                                   const int_tp offY,
                                   viennacl::ocl::context *ctx);
template void greentea_copy<double>(const int_tp N, const cl_mem X,
                                    const int_tp offX, cl_mem Y,
                                    const int_tp offY,
                                    viennacl::ocl::context *ctx);

struct gemm_callback_arg {
  std::vector<cl_event> evs;
  std::vector<cl_mem> imgs;
};

static void CL_CALLBACK gemm_callback (cl_event event,
                                cl_int event_command_exec_status,
                                void *user_data) {
  struct gemm_callback_arg *arg = (struct gemm_callback_arg *) user_data;
  for(int i = 0; i < arg->evs.size(); i++) {
    clReleaseEvent(arg->evs[i]);
  }

  for(int i = 0; i < arg->imgs.size(); i++) {
    clReleaseMemObject(arg->imgs[i]);
  }
  delete arg;
}

// Create and copy buffer to image for GEMM's matrix A and B.
// Will return image to caller if the input image is NULL. Otherwise,
// will use the image directly. It's caller's responsibility to
// release the created image.
void greentea_gpu_gemm_copy_buffer_to_image(int_tp ctx_id,
                 cl_mem *image, cl_mem buffer, int offset,
                 bool is_matrix_a, bool transpose,
                 bool padding, int padded_height,
                 int padded_width, int height,
                 int width, int wait_list_size,
                 cl_event *wait_list,
                 cl_event *event) {

      viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
      viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();
      cl_image_desc desc;
      cl_image_format format;

      memset(&desc, 0, sizeof(desc));
      if (!is_matrix_a && transpose) {
      // For matrix B with transpose, we need to handle them differently.
      // As we can't use the sub group block read to get a row easily,
      // we have to use CL_FLOAT type with read_imagef to get the row.
        cl_int err;
        format.image_channel_data_type = CL_FLOAT;
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        if ( width % 4 == 0 ) {
          desc.image_width = width / 4;
          format.image_channel_order = CL_RGBA;
        } else {
          desc.image_width = width;
          format.image_channel_order = CL_R;
        }
        desc.image_height = height;
        // if (offB == 0 && (desc.image_width % 4) == 0 && N > 8 && K > 8)
        //  desc.mem_object = buffer;
        if (*image == NULL) {
          *image = clCreateImage(
                                ctx.handle().get(),
                                CL_MEM_READ_WRITE,
                                &format,
                                &desc,
                                NULL,
                                &err);
          OCL_CHECK(err);
        }
        // if (!desc.mem_object) {
          size_t origin[] = {0, 0, 0};
          size_t region[] = {(size_t)desc.image_width,
                             (size_t)desc.image_height, 1};
          OCL_CHECK(clEnqueueCopyBufferToImage(ctx.get_queue().handle().get(),
                                     buffer, *image, sizeof(float) * offset,
                                     origin, region, wait_list_size,
                                     wait_list, event));
        // }
        return;
      }

      if (*image == NULL) {
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        format.image_channel_data_type = CL_UNSIGNED_INT8;
        format.image_channel_order = CL_RGBA;
        if (!padding) {
          //if (width % 4 == 0 && offset == 0 && height > 8 && width > 8)
          //  desc.buffer = buffer;
          desc.image_width = width;
          desc.image_height = height;
        } else {
          desc.image_width = padded_width;
          desc.image_height = padded_height;
        }
        cl_int err;
          *image = clCreateImage(ctx.handle().get(),
                              desc.buffer ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE,
                              &format,
                              &desc,
                              NULL,
                              &err);
          OCL_CHECK(err);
      }
      if (!padding && desc.buffer != NULL)
        return;
      if (!padding && desc.buffer == NULL) {
        // copy without padding.
        size_t origin[] = {0, 0, 0};
        size_t region[] = {(size_t)width, (size_t)height, 1};
        OCL_CHECK(clEnqueueCopyBufferToImage(ctx.get_queue().handle().get(),
                                   buffer, *image, sizeof(float) * offset,
                                   origin, region, wait_list_size, wait_list, event));
        return;
      }
      viennacl::ocl::kernel &oclk_gemm_copy = program.get_kernel(
        "gemm_buffer_copy_image_float");

      size_t global_copy[2];
      global_copy[0] = padding ? padded_width : width;
      global_copy[1] = padding ? padded_height : height;
      oclk_gemm_copy.arg(0, WrapHandle(buffer, &ctx));
      oclk_gemm_copy.arg(1, WrapHandle(*image, &ctx));
      oclk_gemm_copy.arg(2, offset);
      oclk_gemm_copy.arg(3, width);
      oclk_gemm_copy.arg(4, height);
      OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                       oclk_gemm_copy.handle().get(),
                                       2, NULL, global_copy, NULL,
                                       wait_list_size, wait_list,
                                       event));
}

// #define GEMM_PROFILING
#ifdef GEMM_PROFILING
#define START_TIMER(n) \
      clFinish(ctx.get_queue().handle().get()); \
      gettimeofday(&start[n], NULL);

#define STOP_TIMER(n) \
      clFinish(ctx.get_queue().handle().get()); \
      gettimeofday(&end[n], NULL);
#else
#define START_TIMER(n)
#define STOP_TIMER(n)
#endif

enum gemm_type_t {
  GEMM_TYPE_NONE = 0,
  GEMM_TYPE_CLBLAS,
  GEMM_TYPE_CLBLAST,
  GEMM_TYPE_VIENNACL,
  GEMM_TYPE_FAST_IMAGE_32_1,
  GEMM_TYPE_FAST_IMAGE_32_2,
  GEMM_TYPE_FAST_BUFFER,
  GEMM_TYPE_MAX
};

static void greentea_gpu_fast_image_gemm(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int_tp M,
                       const int_tp N, const int_tp K, const float alpha,
                       const cl_mem A, const int_tp offA, const cl_mem B,
                       const int_tp offB, const float beta, cl_mem C,
                       const int_tp offC, bool is_image_a, bool is_image_b,
                       enum gemm_type_t gemm_type) {
    CHECK_EQ(gemm_type == GEMM_TYPE_FAST_IMAGE_32_1
             || gemm_type == GEMM_TYPE_FAST_IMAGE_32_2, true)
      << "Invalid fast image gemm type." << std::endl;
    if (is_image_a)
      CHECK_EQ(offA, 0) << "Invalid input image offset." << std::endl;

    if (is_image_b)
      CHECK_EQ(offB, 0) << "Invalid input image offset." << std::endl;

    #ifdef GEMM_PROFILING
    struct timeval start[4], end[4];
    for(int i = 0; i < 4; i++)
      start[i] = end[i];
    #endif
    uint32_t widthA = (TransA == CblasNoTrans) ? K : M;
    uint32_t heightA = (TransA == CblasNoTrans) ? M : K;
    uint32_t widthB = (TransB == CblasNoTrans) ? N : K;
    uint32_t heightB = (TransB == CblasNoTrans) ? K : N;
    // To fix the edge problem casued by the sub group block read.
    // we have to pad the image if it's not multiple of tile.
    // just padding one line is enough as the sub group block read
    // will clamp to edge according to the spec.
    uint32_t padded_k = K + ((K & 7) ? 1 : 0);
    uint32_t imageA_w = (TransA == CblasNoTrans) ? padded_k : M;
    uint32_t imageA_h = (TransA == CblasNoTrans) ? M : padded_k;
    uint32_t imageB_w = (TransB == CblasNoTrans) ? N : padded_k;
    uint32_t imageB_h = (TransB == CblasNoTrans) ? padded_k : N;
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
    viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
                                       ->program();

    cl_mem ImA = NULL;
    cl_mem ImB = NULL;

    cl_event ev[5];
    cl_uint ev_idx = 0;
    memset(ev, 0, sizeof(cl_event) * 5);
    struct gemm_callback_arg * arg = new gemm_callback_arg;
    if (TransB == CblasNoTrans) {
      bool padding_A = false;
      bool padding_B = false;

      if (!is_image_a && !is_image_b) {
        if (M * K < N * K)
          padding_B = true;
        else
          padding_A = true;
      }

      START_TIMER(0);
      if (!is_image_a) {
        greentea_gpu_gemm_copy_buffer_to_image(ctx_id, &ImA, A, offA,
                                  true, TransA != CblasNoTrans,
                                  padding_A, imageA_h, imageA_w,
                                  heightA, widthA, 0, NULL, &ev[ev_idx]);
        if (ev[ev_idx] != NULL)
          ev_idx++;
      }

      STOP_TIMER(0);
      START_TIMER(1);

      if (!is_image_b) {
        greentea_gpu_gemm_copy_buffer_to_image(ctx_id, &ImB, B, offB,
                                  false, false,
                                  padding_B, imageB_h, imageB_w,
                                  heightB, widthB, 0, NULL, &ev[ev_idx]);
        if (ev[ev_idx] != NULL)
          ev_idx++;
      }
      STOP_TIMER(1);
    } else {
      // We will use normal read_imagef to read image B when B has transpose.
      // thus we don't need to pad image A at all.
      START_TIMER(2);
      if (!is_image_a) {
        bool padding;
        padding = !is_image_b;
        greentea_gpu_gemm_copy_buffer_to_image(ctx_id, &ImA, A, offA,
                                  true, TransA != CblasNoTrans,
                                  padding, imageA_h, imageA_w,
                                  heightA, widthA, 0, NULL, &ev[ev_idx]);
        if (ev[ev_idx] != NULL)
          ev_idx++;
      }
      STOP_TIMER(2);
    }
    if (!is_image_a)
      arg->imgs.push_back(ImA);
    else
      ImA = A;
    if (!is_image_b)
      arg->imgs.push_back(ImB);
    else
      ImB = B;

    viennacl::ocl::kernel *oclk_gemm_float;
    std::string kernel_name("gemm_");
    if (gemm_type == GEMM_TYPE_FAST_IMAGE_32_1)
      kernel_name += "32_1_";
    else
      kernel_name += "32_2_";

    if (TransA == CblasNoTrans)
      kernel_name += "N";
    else
      kernel_name += "T";

    if (TransB == CblasNoTrans)
      kernel_name += "N_";
    else {
      kernel_name += "T_";
      if (is_image_b) {
        if (K % 4 == 0)
          kernel_name += "VEC4_";
        else
          kernel_name += "SCALAR_";
      } else {
        kernel_name += "BUFFER_";
      }
    }

    if (alpha == 1)
      kernel_name += "1_";
    else
      kernel_name += "0_";

    if (beta == 0)
      kernel_name += "0";
    else
      kernel_name += "1";
    kernel_name += "_float";

    oclk_gemm_float = &program.get_kernel(kernel_name);

    size_t global[2];

    if (gemm_type == GEMM_TYPE_FAST_IMAGE_32_1)
      global[0] = (size_t)( N + 7 ) & ~7;
    else
      global[0] = (size_t)( (N / 2 ) + 7 ) ^ ~7;

    global[1]  = (size_t)(M + 31) / 32;
    const size_t local[] = {8, 1};

    cl_uint arg_idx = 0;
    oclk_gemm_float->arg(arg_idx++, WrapHandle(ImA, &ctx));
    if (TransB == CblasNoTrans || is_image_b)
      oclk_gemm_float->arg(arg_idx++, WrapHandle(ImB, &ctx));
    else {
      oclk_gemm_float->arg(arg_idx++, WrapHandle(B, &ctx));
      oclk_gemm_float->arg(arg_idx++, offB);
    }
    oclk_gemm_float->arg(arg_idx++, WrapHandle(C, &ctx));
    oclk_gemm_float->arg(arg_idx++, offC);
    oclk_gemm_float->arg(arg_idx++, M);
    oclk_gemm_float->arg(arg_idx++, N);
    oclk_gemm_float->arg(arg_idx++, alpha);
    oclk_gemm_float->arg(arg_idx++, beta);
    oclk_gemm_float->arg(arg_idx++, padded_k);
    if (TransB != CblasNoTrans)
      oclk_gemm_float->arg(arg_idx++, K);

    cl_event *wait_list = NULL;
    if (ev_idx != 0)
      wait_list = &ev[0];
    START_TIMER(3);
    OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                     oclk_gemm_float->handle().get(), 2, NULL,
                                     global, local, ev_idx,
                                     wait_list, &ev[ev_idx]));
    STOP_TIMER(3);
    #ifdef GEMM_PROFILING
    double elapsed[4], total_elapsed;
    for( int i = 0; i < 4; i++ ) {
      elapsed[i] = (end[i].tv_sec - start[i].tv_sec) * 1e6 + (end[i].tv_usec - start[i].tv_usec);
      total_elapsed += elapsed[i];
    }
    printf("kernel name %s \n", kernel_name.c_str());
    printf("gemm %d %d %d %f %f %d %d %f %f %f %f %fGFLOPS %f GFLOPS\n",
            M, K, N, alpha, beta, TransA == CblasNoTrans, TransB == CblasNoTrans,
            elapsed[0] / 1000., elapsed[1] / 1000., elapsed[2] / 1000.,
            elapsed[3] / 1000.,
            M * N * ( 2*K - 1. ) / ( elapsed[3] * 1e3 ),
            M * N * ( 2 * K - 1.) / ( total_elapsed * 1e3 ) );
    #endif
    arg->evs.assign(ev, ev + ev_idx + 1);
    clSetEventCallback(ev[ev_idx], CL_COMPLETE, &gemm_callback, (void*)arg);
}

static void greentea_gpu_fast_buffer_gemm(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int_tp M,
                       const int_tp N, const int_tp K, const float alpha,
                       const cl_mem A, const int_tp offA, const cl_mem B,
                       const int_tp offB, const float beta, cl_mem C,
                       const int_tp offC, enum gemm_type_t gemm_type) {
    CHECK_EQ(gemm_type == GEMM_TYPE_FAST_BUFFER, true)
      << "Invalid fast buffer gemm type." << std::endl;

#ifdef GEMM_PROFILING
    struct timeval start[1], end[1];
    start[0] = end[0];
#endif

    cl_event ev;

    viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
    viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
                                       ->program();
    size_t sub_group_size = 8;
    bool is_small_batch = (M == 2 || M == 4 || M == 8);
    viennacl::ocl::kernel *oclk_gemm_float;
    std::string kernel_name("gemm_buffer_");
    if(TransA == CblasNoTrans && TransB == CblasNoTrans) {
        kernel_name += "NN_float";
    } else if(TransA == CblasNoTrans && TransB != CblasNoTrans) {
        if (M == 2)
          kernel_name +="NT_M_2_float";
        else if (M == 4)
          kernel_name +="NT_M_4_float";
        else if (M == 8)
          kernel_name +="NT_M_8_float";
        else
          kernel_name += "NT_float";
    } else if(TransA != CblasNoTrans && TransB == CblasNoTrans) {
        kernel_name += "TN_float";
    } else {
        kernel_name += "TT_float";
    }
    oclk_gemm_float = &program.get_kernel(kernel_name);
    size_t local[2] = {};
    size_t global[2] = {};
    if (TransA == CblasNoTrans && TransB != CblasNoTrans && is_small_batch ) {
      if(M == 8)
        local[0] = 16;
      else if(M == 4)
        local[0] = 32;
      else
        local[0] = 64;
      local[1] = 1;

      if(M == 8)
        global[0] = N * local[0];
      else
        global[0] = (N + 3) / 4 * local[0];
      global[1] = 1;
    } else {
      size_t lx = sub_group_size;
      size_t ly = (TransB != CblasNoTrans && TransA == CblasNoTrans) ? 16 : 4;
      int dx = (TransB != CblasNoTrans && TransA == CblasNoTrans) ? 1 : 4;
      int dy = 8;
      size_t gx = (size_t)(N + dx - 1) / dx;
      size_t gy = (size_t)(M + dy - 1) / dy;
      global[0] = (gx + lx - 1) / lx * lx;
      global[1] = (gy + ly - 1) / ly * ly;
      local[0] = lx;
      local[1] = ly;
    }

    cl_uint arg_idx = 0;
    oclk_gemm_float->arg(arg_idx++, WrapHandle(A, &ctx));
    oclk_gemm_float->arg(arg_idx++, offA);
    oclk_gemm_float->arg(arg_idx++, WrapHandle(B, &ctx));
    oclk_gemm_float->arg(arg_idx++, offB);
    oclk_gemm_float->arg(arg_idx++, WrapHandle(C, &ctx));
    oclk_gemm_float->arg(arg_idx++, offC);
    oclk_gemm_float->arg(arg_idx++, M);
    oclk_gemm_float->arg(arg_idx++, N);
    oclk_gemm_float->arg(arg_idx++, K);
    oclk_gemm_float->arg(arg_idx++, alpha);
    oclk_gemm_float->arg(arg_idx++, beta);

    START_TIMER(0);
    if(TransB == CblasNoTrans || TransA != CblasNoTrans) {
        int stride = 256;
        for(int start_index = 0; start_index < K; start_index += stride) {
            oclk_gemm_float->arg(arg_idx, start_index);
            OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                            oclk_gemm_float->handle().get(), 2, NULL,
                                            global, local, 0,
                                            NULL, &ev));
        }
    } else {
        OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                         oclk_gemm_float->handle().get(), 2, NULL,
                                         global, local, 0,
                                         NULL, &ev));
    }
    STOP_TIMER(0);
    clReleaseEvent(ev);

#ifdef GEMM_PROFILING
    double total_elapsed;
    total_elapsed = (end[0].tv_sec - start[0].tv_sec) * 1e6 + (end[0].tv_usec - start[0].tv_usec);
    printf("kernel name %s \n", kernel_name.c_str());
    printf("gemm %d %d %d %f %f %d %d %f %fGFLOPS\n",
            M, K, N, alpha, beta, TransA == CblasNoTrans, TransB == CblasNoTrans,
            total_elapsed / 1000., M * N * ( 2 * K - 1.) / ( total_elapsed * 1e3 ) );
#endif
}

template<typename Dtype>
static void greentea_gpu_gemm_common(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int_tp M,
                       const int_tp N, const int_tp K, const Dtype alpha,
                       const cl_mem A, const int_tp offA, const cl_mem B,
                       const int_tp offB, const Dtype beta, cl_mem C,
                       const int_tp offC, bool is_image_a, bool is_image_b,
                       gemm_type_t gemm_type) {

    viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
    int_tp lda = (TransA == CblasNoTrans) ? K : M;
    int_tp ldb = (TransB == CblasNoTrans) ? N : K;
    int_tp ldc = N;

    if (gemm_type == GEMM_TYPE_FAST_IMAGE_32_1 ||
        gemm_type == GEMM_TYPE_FAST_IMAGE_32_2) {
      greentea_gpu_fast_image_gemm(ctx_id, TransA, TransB, M, N, K,
                                   alpha, A, offA, B, offB, beta, C,
                                   offC, is_image_a, is_image_b,
                                   gemm_type);
    } else if (gemm_type == GEMM_TYPE_FAST_BUFFER) {
      greentea_gpu_fast_buffer_gemm(ctx_id, TransA, TransB, M, N, K,
                                    alpha, A, offA, B, offB, beta, C,
                                    offC, gemm_type);
    } else if (gemm_type == GEMM_TYPE_CLBLAS) {
    #if defined(USE_CLBLAS)
      if ((M == 2 || M == 4 || M == 8) && std::is_same<Dtype, float>::value
          && TransA == CblasNoTrans && TransB != CblasNoTrans) {
        greentea_gpu_fast_buffer_gemm(ctx_id, TransA, TransB, M, N, K,
                                      alpha, A, offA, B, offB, beta, C,
                                      offC, GEMM_TYPE_FAST_BUFFER);
      } else {
        clblasOrder clOrder = clblasRowMajor;
        clblasTranspose clTransA =
        (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
        clblasTranspose clTransB =
        (TransB == CblasNoTrans) ? clblasNoTrans : clblasTrans;

        cl_command_queue queue = ctx.get_queue().handle().get();

        if (std::is_same<Dtype, float>::value) {
          GREENTEA_CL_BLAS_CHECK(
              clblasSgemm(clOrder, clTransA, clTransB,
                  M, N, K, alpha, A, offA, lda, B, offB, ldb, beta,
                  C, offC, ldc, 1, &queue, 0, NULL, NULL));
        } else {
          GREENTEA_CL_BLAS_CHECK(
              clblasDgemm(clOrder, clTransA, clTransB,
                  M, N, K, alpha, A, offA, lda, B, offB, ldb, beta,
                  C, offC, ldc, 1, &queue, 0, NULL, NULL));
        }
      }
    #endif
    } else if (gemm_type == GEMM_TYPE_CLBLAST) {
    #ifdef USE_CLBLAST
      cl_command_queue queue = ctx.get_queue().handle().get();

      clblast::Layout layout = clblast::Layout::kRowMajor;
      clblast::Transpose a_transpose = (TransA == CblasNoTrans) ?
        clblast::Transpose::kNo : clblast::Transpose::kYes;
      clblast::Transpose b_transpose = (TransB == CblasNoTrans) ?
        clblast::Transpose::kNo : clblast::Transpose::kYes;

      if (std::is_same<Dtype, float>::value) {
        GREENTEA_CLBLAST_CHECK(
          clblast::Gemm<float>(
            layout, a_transpose, b_transpose,
            M, N, K,
            alpha,
            A, offA, lda,
            B, offB, ldb,
            beta,
            C, offC, ldc,
            &queue));
      } else {
        GREENTEA_CLBLAST_CHECK(
          clblast::Gemm<double>(
            layout, a_transpose, b_transpose,
            M, N, K,
            alpha,
            A, offA, lda,
            B, offB, ldb,
            beta,
            C, offC, ldc,
            &queue));
      }
      #endif
    } else if (gemm_type == GEMM_TYPE_VIENNACL) {
      typedef typename viennacl::matrix_base<Dtype,
          uint_tp, int_tp>::size_type size_type;
      typedef typename viennacl::matrix_base<Dtype,
          uint_tp, int_tp>::size_type difference_type;

      size_type A_size1 = static_cast<size_type>((TransA == CblasTrans) ? K : M);
      size_type A_size2 = static_cast<size_type>((TransA == CblasTrans) ? M : K);

      size_type B_size1 = static_cast<size_type>((TransB == CblasTrans) ? N : K);
      size_type B_size2 = static_cast<size_type>((TransB == CblasTrans) ? K : N);

      viennacl::matrix_base<Dtype, size_t, ptrdiff_t> matA(A, ctx, A_size1,
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(M), A_size2,
                                                       size_type(offA),
                                                       difference_type(1),
                                                       size_type(lda)
                                                       VCL_ROW_MAJOR);

      viennacl::matrix_base<Dtype, size_t, ptrdiff_t> matB(B, ctx, B_size1,
                                                         size_type(0),
                                                         difference_type(1),
                                                         size_type(K), B_size2,
                                                         size_type(offB),
                                                         difference_type(1),
                                                         size_type(ldb)
                                                         VCL_ROW_MAJOR);

      viennacl::matrix_base<Dtype, size_t, ptrdiff_t> matC(C, ctx, size_type(M),
                                                         size_type(0),
                                                         difference_type(1),
                                                         size_type(M),
                                                         size_type(N),
                                                         size_type(offC),
                                                         difference_type(1),
                                                         size_type(ldc)
                                                         VCL_ROW_MAJOR);

      if (TransA == CblasTrans && TransB == CblasTrans)
        viennacl::linalg::prod_impl(viennacl::trans(matA), viennacl::trans(matB),
                                    matC, alpha, beta);
      else if (TransA == CblasTrans && TransB == CblasNoTrans)
        viennacl::linalg::prod_impl(viennacl::trans(matA), matB, matC, alpha,
                                    beta);
      else if (TransA == CblasNoTrans && TransB == CblasTrans)
        viennacl::linalg::prod_impl(matA, viennacl::trans(matB), matC, alpha,
                                  beta);
      else if (TransA == CblasNoTrans && TransB == CblasNoTrans)
        viennacl::linalg::prod_impl(matA, matB, matC, alpha, beta);
   }
}

static void auto_tune_gemm(int ctx_id, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_TRANSPOSE TransB,
                    gemm_type_t *tuned_gemm_types,
                    bool use_fast_gemm_image) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  int M = 1024;
  int K = 512;
  int N = 1024;
  cl_int err;
  cl_mem A = clCreateBuffer(ctx.handle().get(), CL_MEM_ALLOC_HOST_PTR, M * K * sizeof(float), NULL, &err);
  OCL_CHECK(err);
  cl_mem B = clCreateBuffer(ctx.handle().get(), CL_MEM_ALLOC_HOST_PTR, K * N * sizeof(float), NULL, &err);
  OCL_CHECK(err);
  cl_mem C = clCreateBuffer(ctx.handle().get(), CL_MEM_ALLOC_HOST_PTR, M * N * sizeof(float), NULL, &err);
  OCL_CHECK(err);

  std::vector<gemm_type_t> gemm_tests;

  gemm_tests.push_back(GEMM_TYPE_VIENNACL);
  if(use_fast_gemm_image)
    gemm_tests.push_back(GEMM_TYPE_FAST_IMAGE_32_1);
  gemm_tests.push_back(GEMM_TYPE_FAST_BUFFER);

#ifdef USE_CLBLAS
  gemm_tests.push_back(GEMM_TYPE_CLBLAS);
#endif
#ifdef USE_CLBLAST
  gemm_tests.push_back(GEMM_TYPE_CLBLAST);
#endif
  // warm up.
  for( int i = 0; i < gemm_tests.size(); i++ ) {
    greentea_gpu_gemm_common(ctx_id, TransA, TransB, M, N, K,
                             1.0f, A, 0, B, 0, 0.0f, C, 0, false, false,
                             gemm_tests[i]);
  }
  float fastest_time = 1e10;
  int fastest_index = -1;
  clFinish(ctx.get_queue().handle().get());
  for( int i = 0; i < gemm_tests.size(); i++ ) {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    greentea_gpu_gemm_common(ctx_id, TransA, TransB, M, N, K,
                             1.0f, A, 0, B, 0, 0.0f, C, 0, false, false,
                             gemm_tests[i]);
    clFinish(ctx.get_queue().handle().get());
    gettimeofday(&end, NULL);
    float elapsed = (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
    if (elapsed < fastest_time) {
      fastest_time = elapsed;
      fastest_index = i;
    }
  }
  clReleaseMemObject(A);
  clReleaseMemObject(B);
  clReleaseMemObject(C);

  if (fastest_index >= 0) {
    tuned_gemm_types[ctx_id] = gemm_tests[fastest_index];
#ifdef GEMM_PROFILING
    printf("The tuned GEMM kernel get %f GFLOPS with kernel type %d.\n",
            M*N*(2*(double)K-1)/(fastest_time * 1e3),
            tuned_gemm_types[ctx_id]);
#endif
  }
}

static gemm_type_t tuned_gemm_nn_types_with_image[16] = {GEMM_TYPE_NONE};
static gemm_type_t tuned_gemm_nt_types_with_image[16] = {GEMM_TYPE_NONE};
static gemm_type_t tuned_gemm_tn_types_with_image[16] = {GEMM_TYPE_NONE};
static gemm_type_t tuned_gemm_tt_types_with_image[16] = {GEMM_TYPE_NONE};

static gemm_type_t tuned_gemm_nn_types_without_image[16] = {GEMM_TYPE_NONE};
static gemm_type_t tuned_gemm_nt_types_without_image[16] = {GEMM_TYPE_NONE};
static gemm_type_t tuned_gemm_tn_types_without_image[16] = {GEMM_TYPE_NONE};
static gemm_type_t tuned_gemm_tt_types_without_image[16] = {GEMM_TYPE_NONE};

static void auto_tune_gemm_all(int ctx_id, bool use_fast_gemm_image) {
  if(use_fast_gemm_image) {
    auto_tune_gemm(ctx_id, CblasNoTrans, CblasNoTrans, tuned_gemm_nn_types_with_image, true);
    auto_tune_gemm(ctx_id, CblasNoTrans, CblasTrans, tuned_gemm_nt_types_with_image, true);
    auto_tune_gemm(ctx_id, CblasTrans, CblasNoTrans, tuned_gemm_tn_types_with_image, true);
    auto_tune_gemm(ctx_id, CblasTrans, CblasTrans, tuned_gemm_tt_types_with_image, true);
  } else {
    auto_tune_gemm(ctx_id, CblasNoTrans, CblasNoTrans, tuned_gemm_nn_types_without_image, false);
    auto_tune_gemm(ctx_id, CblasNoTrans, CblasTrans, tuned_gemm_nt_types_without_image, false);
    auto_tune_gemm(ctx_id, CblasTrans, CblasNoTrans, tuned_gemm_tn_types_without_image, false);
    auto_tune_gemm(ctx_id, CblasTrans, CblasTrans, tuned_gemm_tt_types_without_image, false);
  }
}

static boost::mutex auto_tune_gemm_mutex;

template<typename Dtype>
void greentea_gpu_gemm(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int_tp M,
                       const int_tp N, const int_tp K, const Dtype alpha,
                       const cl_mem A, const int_tp offA, const cl_mem B,
                       const int_tp offB, const Dtype beta, cl_mem C,
                       const int_tp offC, bool is_image_a, bool is_image_b) {
  CHECK_LT(ctx_id, 16) << "Too many GPU devices.";
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  bool use_fast_gemm_image = false;
  bool use_fast_gemm_buffer = false;

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    Dtype* Aptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), A, true, CL_MAP_READ,
        sizeof(Dtype) * offA, sizeof(Dtype) * M * K, 0, NULL, NULL, NULL));
    Dtype* Bptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), B, true, CL_MAP_READ,
        sizeof(Dtype) * offB, sizeof(Dtype) * N * K, 0, NULL, NULL, NULL));
    Dtype* Cptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), C, true, CL_MAP_READ | CL_MAP_WRITE,
        sizeof(Dtype) * offC, sizeof(Dtype) * M * N, 0, NULL, NULL, NULL));

    caffe_cpu_gemm<Dtype>(TransA, TransB, M, N, K, alpha, Aptr, Bptr, beta,
                          Cptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), A, Aptr, 0, NULL,
    NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), B, Bptr, 0, NULL,
    NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), C, Cptr, 0, NULL,
    NULL);
    return;
  }

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_GPU &&
     std::is_same<Dtype, float>::value) {
    // Check whether can/should we use the fast gemm driver.
    // There are the following considerations/restrications:
    // 1. The fast gemm kernel is using image which has a size limitation.
    // 2. The fast gemm kernel is using the intel sub group extension.
    // 3. Currently, only the IGC compiler (the driver version is 16.xxx)
    //    can get better performance with the fast gemm.
    // Cap at 1 MB to capture faulty OpenCL implementations (nVidia)
    bool has_sub_group_ext = ctx.devices()[0].extensions().find("cl_intel_subgroups")
                               != std::string::npos;
    if (has_sub_group_ext) {
      size_t max_image_size = std::min(ctx.devices()[0].image2d_max_width(),
                                       ctx.devices()[0].image2d_max_height());
      if (M <= max_image_size &&
          K <= max_image_size &&
          N <= max_image_size) {
        use_fast_gemm_image = true;
      }
      use_fast_gemm_buffer = true;
    }
  }

  gemm_type_t preferred_gemm_type = GEMM_TYPE_VIENNACL;
#ifdef USE_CLBLAS
  preferred_gemm_type = GEMM_TYPE_CLBLAS;
#endif
#ifdef  USE_CLBLAST
  preferred_gemm_type = GEMM_TYPE_CLBLAST;
#endif

  {
    boost::mutex::scoped_lock lock(auto_tune_gemm_mutex);
    if(use_fast_gemm_image) {
      if (tuned_gemm_nn_types_with_image[ctx_id] == GEMM_TYPE_NONE) {
        auto_tune_gemm_all(ctx_id, true);
      }

      if (TransA == CblasNoTrans && TransB == CblasNoTrans)
        preferred_gemm_type = tuned_gemm_nn_types_with_image[ctx_id];
      else if (TransA == CblasTrans && TransB == CblasNoTrans)
        preferred_gemm_type = tuned_gemm_tn_types_with_image[ctx_id];
      else if (TransA == CblasNoTrans && TransB == CblasTrans)
        preferred_gemm_type = tuned_gemm_nt_types_with_image[ctx_id];
      else if (TransA == CblasTrans && TransB == CblasTrans)
        preferred_gemm_type = tuned_gemm_tt_types_with_image[ctx_id];
    } else if(use_fast_gemm_buffer) {
      if (tuned_gemm_nn_types_without_image[ctx_id] == GEMM_TYPE_NONE) {
        auto_tune_gemm_all(ctx_id, false);
      }

      if (TransA == CblasNoTrans && TransB == CblasNoTrans)
        preferred_gemm_type = tuned_gemm_nn_types_without_image[ctx_id];
      else if (TransA == CblasTrans && TransB == CblasNoTrans)
        preferred_gemm_type = tuned_gemm_tn_types_without_image[ctx_id];
      else if (TransA == CblasNoTrans && TransB == CblasTrans)
        preferred_gemm_type = tuned_gemm_nt_types_without_image[ctx_id];
      else if (TransA == CblasTrans && TransB == CblasTrans)
        preferred_gemm_type = tuned_gemm_tt_types_without_image[ctx_id];
    }
  }

  CHECK_EQ(use_fast_gemm_image || (!is_image_a && !is_image_b), true)
    << "Invalid GEMM parameters.";

  if (is_image_a || is_image_b)
    preferred_gemm_type = GEMM_TYPE_FAST_IMAGE_32_1;

  greentea_gpu_gemm_common(ctx_id, TransA, TransB, M, N, K, alpha, A, offA,
                           B, offB, beta, C, offC, is_image_a, is_image_b,
                           preferred_gemm_type);
}

template void greentea_gpu_gemm<float>(const int_tp ctx_id,
                                       const CBLAS_TRANSPOSE TransA,
                                       const CBLAS_TRANSPOSE TransB,
                                       const int_tp M, const int_tp N,
                                       const int_tp K, const float alpha,
                                       const cl_mem A, const int_tp offA,
                                       const cl_mem B, const int_tp offB,
                                       const float beta, cl_mem C,
                                       const int_tp offC,
                                       const bool is_image_a = false,
                                       const bool is_image_b = false);
template void greentea_gpu_gemm<double>(const int_tp ctx_id,
                                        const CBLAS_TRANSPOSE TransA,
                                        const CBLAS_TRANSPOSE TransB,
                                        const int_tp M, const int_tp N,
                                        const int_tp K, const double alpha,
                                        const cl_mem A, const int_tp offA,
                                        const cl_mem B, const int_tp offB,
                                        const double beta, cl_mem C,
                                        const int_tp offC,
                                        const bool is_image_a = false,
                                        const bool is_image_b = false);

template void greentea_gpu_gemm_common<float>(const int_tp ctx_id,
                                       const CBLAS_TRANSPOSE TransA,
                                       const CBLAS_TRANSPOSE TransB,
                                       const int_tp M, const int_tp N,
                                       const int_tp K, const float alpha,
                                       const cl_mem A, const int_tp offA,
                                       const cl_mem B, const int_tp offB,
                                       const float beta, cl_mem C,
                                       const int_tp offC,
                                       const bool is_image_a,
                                       const bool is_image_b,
                                       const gemm_type_t);
template void greentea_gpu_gemm_common<double>(const int_tp ctx_id,
                                        const CBLAS_TRANSPOSE TransA,
                                        const CBLAS_TRANSPOSE TransB,
                                        const int_tp M, const int_tp N,
                                        const int_tp K, const double alpha,
                                        const cl_mem A, const int_tp offA,
                                        const cl_mem B, const int_tp offB,
                                        const double beta, cl_mem C,
                                        const int_tp offC,
                                        const bool is_image_a,
                                        const bool is_image_b,
                                        const gemm_type_t);

template<typename Dtype>
void greentea_gpu_gemv(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const int_tp M, const int_tp N, const Dtype alpha,
                       const cl_mem A, const int_tp offA, const cl_mem x,
                       const int_tp offx, const Dtype beta, cl_mem y,
                       const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    Dtype* Aptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), A, true, CL_MAP_READ,
        sizeof(Dtype) * offA, sizeof(Dtype) * M * N, 0, NULL, NULL, NULL));
    Dtype* xptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x, true, CL_MAP_READ,
        sizeof(Dtype) * offx, sizeof(Dtype) * (TransA == CblasTrans) ? M : N, 0,
        NULL,
        NULL, NULL));
    Dtype* yptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), y, true, CL_MAP_READ | CL_MAP_WRITE,
        sizeof(Dtype) * offy, sizeof(Dtype) * (TransA == CblasTrans) ? N : M, 0,
        NULL,
        NULL, NULL));

    caffe_cpu_gemv<Dtype>(TransA, M, N, alpha, Aptr, xptr, beta, yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), A, Aptr, 0, NULL,
    NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x, xptr, 0, NULL,
    NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), y, yptr, 0, NULL,
    NULL);
  } else {
      if (std::is_same<Dtype, float>::value && TransA == CblasNoTrans) {
        viennacl::ocl::program &program =
            (Caffe::Get().GetDevice(ctx_id, false))
                  ->program();
        viennacl::ocl::kernel &k =
            program.get_kernel(CL_KERNEL_SELECT("matvec_mul4"));
        uint row_size = M;
        uint col_size = N;
        size_t localsize = 128;
        size_t globalsize = row_size / 4 * localsize;

        uint argId = 0;
        k.arg(argId++, WrapHandle(A, &ctx));
        k.arg(argId++, offA);
        k.arg(argId++, cl_uint(col_size));
        k.arg(argId++, cl_uint(col_size%4));
        k.arg(argId++, WrapHandle(x, &ctx));
        k.arg(argId++, offx);
        k.arg(argId++, alpha);
        k.arg(argId++, beta);
        k.arg(argId++, WrapHandle(y, &ctx));
        k.arg(argId++, offy);
        k.arg(argId++, viennacl::ocl::local_mem(sizeof(cl_float4) * localsize));

        clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                     k.handle().get(), 1,
                                     NULL,
                                     &globalsize,
                                     &localsize, 0, NULL,
                                     NULL);
        if ((row_size % 4) != 0) {
          viennacl::ocl::kernel &k_1 =
              program.get_kernel(CL_KERNEL_SELECT("matvec_mul1"));
          size_t localsize = 128;
          size_t globalsize = row_size % 4 * localsize;
          uint row_offset = row_size - (row_size % 4);

          uint argId = 0;
          k_1.arg(argId++, WrapHandle(A, &ctx));
          k_1.arg(argId++, offA);
          k_1.arg(argId++, cl_uint(col_size));
          k_1.arg(argId++, cl_uint(row_offset));
          k_1.arg(argId++, cl_uint(col_size%4));
          k_1.arg(argId++, WrapHandle(x, &ctx));
          k_1.arg(argId++, offx);
          k_1.arg(argId++, alpha);
          k_1.arg(argId++, beta);
          k_1.arg(argId++, WrapHandle(y, &ctx));
          k_1.arg(argId++, offy);
          k_1.arg(argId++,
                  viennacl::ocl::local_mem(sizeof(cl_float) * localsize));

          clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                       k_1.handle().get(), 1,
                                       NULL,
                                       &globalsize,
                                       &localsize, 0, NULL,
                                       NULL);
        }
      } else {
#if defined(USE_CLBLAS)

        clblasTranspose clTransA =
        (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;

        cl_command_queue queue = ctx.get_queue().handle().get();

        if (std::is_same<Dtype, float>::value) {
          GREENTEA_CL_BLAS_CHECK(
              clblasSgemv(clblasRowMajor,
                  clTransA, M, N, alpha, A, offA, N, x, offx, 1,
                  beta, y, offy, 1, 1, &queue, 0, NULL, NULL));
        } else {
          GREENTEA_CL_BLAS_CHECK(
              clblasDgemv(clblasRowMajor,
                  clTransA, M, N, alpha, A, offA, N, x, offx, 1,
                  beta, y, offy, 1, 1, &queue, 0, NULL, NULL));
        }

#elif defined(USE_CLBLAST)

        cl_command_queue queue = ctx.get_queue().handle().get();

        clblast::Layout layout = clblast::Layout::kRowMajor;
        clblast::Transpose a_transpose = (TransA == CblasNoTrans) ?
          clblast::Transpose::kNo : clblast::Transpose::kYes;

        const size_t ldA = N;
        const size_t incx = 1;
        const size_t incy = 1;

        if (std::is_same<Dtype, float>::value) {
          GREENTEA_CLBLAST_CHECK(
            clblast::Gemv<float>(
              layout, a_transpose,
              M, N,
              alpha,
              A, offA, ldA,
              x, offx, incx,
              beta,
              y, offy, incy,
              &queue));
        } else {
          GREENTEA_CLBLAST_CHECK(
            clblast::Gemv<double>(
              layout, a_transpose,
              M, N,
              alpha,
              A, offA, ldA,
              x, offx, incx,
              beta,
              y, offy, incy,
              &queue));
        }

#else  // default (ViennaCL)

        typedef typename viennacl::vector_base<Dtype,
            uint_tp, int_tp>::size_type size_type;
        typedef typename viennacl::vector_base<Dtype,
            uint_tp, int_tp>::size_type difference_type;

        viennacl::vector_base<Dtype, size_t, ptrdiff_t> v1(
            x, size_type((TransA == CblasTrans) ? M : N), size_type(offx),
            difference_type(1), ctx);
        viennacl::vector_base<Dtype, size_t, ptrdiff_t> v2(
            y, size_type((TransA == CblasTrans) ? N : M), size_type(offy),
            difference_type(1), ctx);
        viennacl::matrix_base<Dtype, size_t, ptrdiff_t> mat(
                                                        A, ctx, size_type(M),
                                                          size_type(0),
                                                          difference_type(1),
                                                          size_type(M),
                                                          size_type(N),
                                                          size_type(offA),
                                                          difference_type(1),
                                                          size_type(N)
                                                          VCL_ROW_MAJOR);
        v2 *= beta;
        if (TransA == CblasTrans) {
          v2 += alpha * viennacl::linalg::prod(viennacl::trans(mat), v1);
        } else {
          v2 += alpha * viennacl::linalg::prod(mat, v1);
        }

#endif  // clBLAS, CLBlast, or default (ViennaCL)
     }
  }
}

template void greentea_gpu_gemv<float>(const int_tp ctx_id,
                                       const CBLAS_TRANSPOSE TransA,
                                       const int_tp M, const int_tp N,
                                       const float alpha, const cl_mem A,
                                       const int_tp offA, const cl_mem x,
                                       const int_tp offx, const float beta,
                                       cl_mem y, const int_tp offy);
template void greentea_gpu_gemv<double>(const int_tp ctx_id,
                                        const CBLAS_TRANSPOSE TransA,
                                        const int_tp M, const int_tp N,
                                        const double alpha, const cl_mem A,
                                        const int_tp offA, const cl_mem x,
                                        const int_tp offx, const double beta,
                                        cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_axpy(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                       const cl_mem X, const int_tp offX, cl_mem Y,
                       const int_tp offY) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    Dtype* Xptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X, true, CL_MAP_READ,
        sizeof(Dtype) * offX, sizeof(Dtype) * N, 0, NULL, NULL, NULL));
    Dtype* Yptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Y, true, CL_MAP_WRITE,
        sizeof(Dtype) * offY, sizeof(Dtype) * N, 0, NULL, NULL, NULL));

    caffe_axpy<Dtype>(N, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X, Xptr, 0, NULL,
    NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Y, Yptr, 0, NULL,
    NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasSaxpy(N, alpha, X, offX,
              1, Y, offY, 1, 1, &queue, 0, NULL, NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDaxpy(N, alpha, X, offX,
              1, Y, offY, 1, 1, &queue, 0, NULL, NULL));
    }

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CLBLAST_CHECK(
        clblast::Axpy<float>(
          N,
          alpha,
          X, offX, incX,
          Y, offY, incY,
          &queue));
    } else {
      GREENTEA_CLBLAST_CHECK(
        clblast::Axpy<double>(
          N,
          alpha,
          X, offX, incX,
          Y, offY, incY,
          &queue));
    }

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v1(X, size_type(N),
                                                     size_type(offX),
                                                     difference_type(1), ctx);
    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v2(Y, size_type(N),
                                                     size_type(offY),
                                                     difference_type(1), ctx);
    v2 += alpha * v1;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template void greentea_gpu_axpy<float>(const int_tp ctx_id, const int_tp N,
                                       const float alpha, const cl_mem X,
                                       const int_tp offX, cl_mem Y,
                                       const int_tp offY);
template void greentea_gpu_axpy<double>(const int_tp ctx_id, const int_tp N,
                                        const double alpha, const cl_mem X,
                                        const int_tp offX, cl_mem Y,
                                        const int_tp offY);

template<typename Dtype>
void greentea_gpu_mul(const int_tp ctx_id, const int_tp N, const cl_mem a,
                      const int_tp offa, const cl_mem b, const int_tp offb,
                      cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_mul = program.get_kernel(CL_KERNEL_SELECT("mul"));
  viennacl::ocl::enqueue(
      oclk_mul(N, WrapHandle(a, &ctx), offa, WrapHandle(b, &ctx), offb,
               WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_mul<float>(const int_tp ctx_id, const int_tp N,
                                      const cl_mem a, const int_tp offa,
                                      const cl_mem b, const int_tp offb,
                                      cl_mem y, const int_tp offy);
template void greentea_gpu_mul<double>(const int_tp ctx_id, const int_tp N,
                                       const cl_mem a, const int_tp offa,
                                       const cl_mem b, const int_tp offb,
                                       cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_div(const int_tp ctx_id, const int_tp N, const cl_mem a,
                      const int_tp offa, const cl_mem b, const int_tp offb,
                      cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_div = program.get_kernel(CL_KERNEL_SELECT("div"));
  viennacl::ocl::enqueue(
      oclk_div(N, WrapHandle(a, &ctx), offa, WrapHandle(b, &ctx), offb,
               WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_div<float>(const int_tp ctx_id, const int_tp N,
                                      const cl_mem a, const int_tp offa,
                                      const cl_mem b, const int_tp offb,
                                      cl_mem y, const int_tp offy);
template void greentea_gpu_div<double>(const int_tp ctx_id, const int_tp N,
                                       const cl_mem a, const int_tp offa,
                                       const cl_mem b, const int_tp offb,
                                       cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_scal(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                       cl_mem x, int_tp offx) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    Dtype* xptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x, true, CL_MAP_READ | CL_MAP_WRITE,
        sizeof(Dtype) * offx, sizeof(Dtype) * N, 0, NULL, NULL, NULL));

    caffe_scal<Dtype>(N, alpha, xptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x, xptr, 0, NULL,
    NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(clblasSscal(N, alpha, x, offx,
              1, 1, &queue, 0, NULL, NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(clblasDscal(N, alpha, x, offx,
              1, 1, &queue, 0, NULL, NULL));
    }

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incx = 1;

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CLBLAST_CHECK(
        clblast::Scal<float>(
          N,
          alpha,
          x, offx, incx,
          &queue));
    } else {
      GREENTEA_CLBLAST_CHECK(
        clblast::Scal<double>(
          N,
          alpha,
          x, offx, incx,
          &queue));
    }

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v1(x, size_type(N),
                                                     size_type(offx),
                                                     difference_type(1), ctx);
    v1 *= alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template void greentea_gpu_scal<float>(const int_tp ctx_id, const int_tp N,
                                       const float alpha, cl_mem x,
                                       const int_tp offx);
template void greentea_gpu_scal<double>(const int_tp ctx_id, const int_tp N,
                                        const double alpha, cl_mem x,
                                        const int_tp offx);

template<typename Dtype>
void greentea_gpu_axpby(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                        const cl_mem X, const int_tp offX, const Dtype beta,
                        cl_mem Y, const int_tp offY) {
  greentea_gpu_scal<Dtype>(ctx_id, N, beta, Y, offY);
  greentea_gpu_axpy<Dtype>(ctx_id, N, alpha, X, offX, Y, offY);
}

template void greentea_gpu_axpby<float>(const int_tp ctx_id, const int_tp N,
                                        const float alpha, const cl_mem X,
                                        const int_tp offX, const float beta,
                                        cl_mem Y, const int_tp offY);

template void greentea_gpu_axpby<double>(const int_tp ctx_id, const int_tp N,
                                         const double alpha, const cl_mem X,
                                         const int_tp offX, const double beta,
                                         cl_mem Y, const int_tp offY);

template<typename Dtype>
void greentea_gpu_dot(const int_tp ctx_id, const int_tp n, const cl_mem X,
                      const int_tp offX, const cl_mem Y, const int_tp offY,
                      Dtype* out) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    Dtype* Xptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X, true, CL_MAP_READ,
        sizeof(Dtype) * offX, sizeof(Dtype) * n, 0, NULL, NULL, NULL));
    Dtype* Yptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Y, true, CL_MAP_READ,
        sizeof(Dtype) * offY, sizeof(Dtype) * n, 0, NULL, NULL, NULL));

    *out = caffe_cpu_dot<Dtype>(n, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X, Xptr, 0, NULL,
    NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Y, Yptr, 0, NULL,
    NULL);

  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err;
    cl_mem gpuout = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
        sizeof(Dtype), NULL, &err);
    cl_mem scratch = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
        n * sizeof(Dtype), NULL, &err);

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasSdot(n, gpuout, 0, X, offX, 1, Y,
              offY, 1, scratch, 1, &queue, 0, NULL, NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDdot(n, gpuout, 0, X, offX, 1, Y,
              offY, 1, scratch, 1, &queue, 0, NULL, NULL));
    }

    greentea_gpu_memcpy(sizeof(Dtype), gpuout, 0, out, &ctx);

    clReleaseMemObject(gpuout);
    clReleaseMemObject(scratch);

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err = CL_SUCCESS;
    cl_mem Z = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      sizeof(Dtype), NULL, &err);
    // TODO: error handling.

    const size_t offZ = 0;
    const size_t incX = 1;
    const size_t incY = 1;

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CLBLAST_CHECK(
        clblast::Dot<float>(
          n,
          Z, offZ,
          X, offX, incX,
          Y, offY, incY,
          &queue));
    } else {
      GREENTEA_CLBLAST_CHECK(
        clblast::Dot<double>(
          n,
          Z, offZ,
          X, offX, incX,
          Y, offY, incY,
          &queue));
    }

    greentea_gpu_memcpy(sizeof(Dtype), Z, offZ, out, &ctx);
    clReleaseMemObject(Z);

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v1(X, size_type(n),
                                                     size_type(offX),
                                                     difference_type(1), ctx);
    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v2(Y, size_type(n),
                                                     size_type(offY),
                                                     difference_type(1), ctx);

    *out = viennacl::linalg::inner_prod(v1, v2);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template void greentea_gpu_dot<float>(const int_tp ctx_id, const int_tp n,
                                      const cl_mem X, const int_tp offX,
                                      const cl_mem Y, const int_tp offY,
                                      float* out);
template void greentea_gpu_dot<double>(const int_tp ctx_id, const int_tp n,
                                       const cl_mem X, const int_tp offX,
                                       const cl_mem Y, const int_tp offY,
                                       double* out);

template<typename Dtype>
void greentea_gpu_asum(const int_tp ctx_id, const int_tp n, const cl_mem X,
                       const int_tp offX, Dtype* Y) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    Dtype* Xptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X, true, CL_MAP_READ,
        sizeof(Dtype) * offX, sizeof(Dtype) * n, 0, NULL, NULL, NULL));

    *Y = caffe_cpu_asum<Dtype>(n, Xptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X, Xptr, 0, NULL,
    NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err;
    cl_mem gpuout = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
        sizeof(Dtype), NULL, &err);
    cl_mem scratch = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
        n * sizeof(Dtype), NULL, &err);

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasSasum(n, gpuout, 0, X, offX, 1,
              scratch, 1, &queue, 0, NULL, NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDasum(n, gpuout, 0, X, offX, 1,
              scratch, 1, &queue, 0, NULL, NULL));
    }

    greentea_gpu_memcpy(sizeof(Dtype), gpuout, 0, Y, &ctx);

    clReleaseMemObject(gpuout);
    clReleaseMemObject(scratch);

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err = CL_SUCCESS;
    cl_mem Z = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      sizeof(Dtype), NULL, &err);
    // TODO: error handling.

    const size_t offZ = 0;
    const size_t incX = 1;

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CLBLAST_CHECK(
        clblast::Asum<float>(
          n,
          Z, offZ,
          X, offX, incX,
          &queue));
    } else {
      GREENTEA_CLBLAST_CHECK(
        clblast::Asum<double>(
          n,
          Z, offZ,
          X, offX, incX,
          &queue));
    }

    greentea_gpu_memcpy(sizeof(Dtype), Z, offZ, Y, &ctx);

    clReleaseMemObject(Z);

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v1(X, size_type(n),
                                                     size_type(offX),
                                                     difference_type(1), ctx);

    *Y = viennacl::linalg::norm_1(v1);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template void greentea_gpu_asum<float>(const int_tp ctx_id, const int_tp n,
                                       const cl_mem X, const int_tp offX,
                                       float* Y);
template void greentea_gpu_asum<double>(const int_tp ctx_id, const int_tp n,
                                        const cl_mem X, const int_tp offX,
                                        double* Y);

template<typename Dtype>
void greentea_gpu_scale(const int_tp ctx_id, const int_tp n, const Dtype alpha,
                        const cl_mem X, const int_tp offX, cl_mem Y,
                        const int_tp offY) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    Dtype* Xptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X, true, CL_MAP_READ,
        sizeof(Dtype) * offX, sizeof(Dtype) * n, 0, NULL, NULL, NULL));
    Dtype* Yptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Y, true, CL_MAP_WRITE,
        sizeof(Dtype) * offY, sizeof(Dtype) * n, 0, NULL, NULL, NULL));

    caffe_cpu_scale<Dtype>(n, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X, Xptr, 0, NULL,
    NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Y, Yptr, 0, NULL,
    NULL);
  } else {
#if defined(USE_CLBLAS)

    // FIXME: Remove, as can reuse ctx obtained above?
    viennacl::ocl::context ctx = viennacl::ocl::get_context(ctx_id);
    cl_command_queue queue = ctx.get_queue().handle().get();

    // FIXME: Use xAXPY with beta = 0?
    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasScopy(n, X, offX, 1, Y, offY, 1, 1, &queue, 0, NULL, NULL));
      GREENTEA_CL_BLAS_CHECK(
          clblasSscal(n, alpha, Y, offY, 1, 1, &queue, 0, NULL, NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDcopy(n, X, offX, 1, Y, offY, 1, 1, &queue, 0, NULL, NULL));
      GREENTEA_CL_BLAS_CHECK(
          clblasDscal(n, alpha, Y, offY, 1, 1, &queue, 0, NULL, NULL));
    }

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CLBLAST_CHECK(
        clblast::Copy<float>(
          n,
          X, offX, incX,
          Y, offY, incY,
          &queue));
      GREENTEA_CLBLAST_CHECK(
        clblast::Scal<float>(
          n,
          alpha,
          Y, offY, incY,
          &queue));
    } else {
      GREENTEA_CLBLAST_CHECK(
        clblast::Copy<double>(
          n,
          X, offX, incX,
          Y, offY, incY,
          &queue));
      GREENTEA_CLBLAST_CHECK(
        clblast::Scal<double>(
          n,
          alpha,
          Y, offY, incY,
          &queue));
    }

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v1(X, size_type(n),
                                                     size_type(offX),
                                                     difference_type(1), ctx);
    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v2(Y, size_type(n),
                                                     size_type(offY),
                                                     difference_type(1), ctx);

    v2 = v1 * alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template void greentea_gpu_scale<float>(const int_tp ctx_id, const int_tp n,
                                        const float alpha, const cl_mem X,
                                        const int_tp offX, cl_mem Y,
                                        const int_tp offY);

template void greentea_gpu_scale<double>(const int_tp ctx_id, const int_tp n,
                                         const double alpha, const cl_mem X,
                                         const int_tp offX, cl_mem Y,
                                         const int_tp offY);

template<typename Dtype>
void greentea_gpu_set(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                      cl_mem Y, const int_tp offY) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();
  // OpenCL Version >= 1.2 approach
  // clEnqueueFillBuffer(ctx.get_queue().handle().get(),
  //                  Y, &alpha, sizeof(Dtype),
  //                  offY, N, 0, NULL, NULL);

  // OpenCL Version < 1.2 fallback
  viennacl::ocl::kernel &oclk_fill = program.get_kernel(
      CL_KERNEL_SELECT("fill"));
  viennacl::ocl::enqueue(oclk_fill(N, alpha, WrapHandle(Y, &ctx), offY),
                         ctx.get_queue());
}

template void greentea_gpu_set<int_tp>(const int_tp ctx_id, const int_tp N,
                                       const int_tp alpha, cl_mem Y,
                                       const int_tp offY);
template void greentea_gpu_set<float>(const int_tp ctx_id, const int_tp N,
                                      const float alpha, cl_mem Y,
                                      const int_tp offY);
template void greentea_gpu_set<double>(const int_tp ctx_id, const int_tp N,
                                       const double alpha, cl_mem Y,
                                       const int_tp offY);

template<typename Dtype>
void greentea_gpu_add_scalar(const int_tp ctx_id, const int_tp N,
                             const Dtype alpha, cl_mem Y, const int_tp offY) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_add_scalar = program.get_kernel(
      CL_KERNEL_SELECT("add_scalar"));
  viennacl::ocl::enqueue(oclk_add_scalar(N, alpha, WrapHandle(Y, &ctx), offY),
                         ctx.get_queue());
}

template void greentea_gpu_add_scalar<float>(const int_tp ctx_id,
                                             const int_tp N, const float alpha,
                                             cl_mem Y, const int_tp offY);
template void greentea_gpu_add_scalar<double>(const int_tp ctx_id,
                                              const int_tp N,
                                              const double alpha, cl_mem Y,
                                              const int_tp offY);

template<typename Dtype>
void greentea_gpu_add(const int_tp ctx_id, const int_tp n, const cl_mem a,
                      const int_tp offa, const cl_mem b, const int_tp offb,
                      cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_add = program.get_kernel(CL_KERNEL_SELECT("add"));
  viennacl::ocl::enqueue(
      oclk_add(n, WrapHandle(a, &ctx), offa, WrapHandle(b, &ctx), offb,
               WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_add<float>(const int_tp ctx_id, const int_tp n,
                                      const cl_mem a, const int_tp offa,
                                      const cl_mem b, const int_tp offb,
                                      cl_mem y, const int_tp offy);
template void greentea_gpu_add<double>(const int_tp ctx_id, const int_tp n,
                                       const cl_mem a, const int_tp offa,
                                       const cl_mem b, const int_tp offb,
                                       cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_sub(const int_tp ctx_id, const int_tp n, const cl_mem a,
                      const int_tp offa, const cl_mem b, const int_tp offb,
                      cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_sub = program.get_kernel(CL_KERNEL_SELECT("sub"));
  viennacl::ocl::enqueue(
      oclk_sub(n, WrapHandle(a, &ctx), offa, WrapHandle(b, &ctx), offb,
               WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_sub<float>(const int_tp ctx_id, const int_tp n,
                                      const cl_mem a, const int_tp offa,
                                      const cl_mem b, const int_tp offb,
                                      cl_mem y, const int_tp offy);
template void greentea_gpu_sub<double>(const int_tp ctx_id, const int_tp n,
                                       const cl_mem a, const int_tp offa,
                                       const cl_mem b, const int_tp offb,
                                       cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_abs(const int_tp ctx_id, const int_tp N, const cl_mem a,
                      const int_tp offa, cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_abs = program.get_kernel(CL_KERNEL_SELECT("abs"));
  viennacl::ocl::enqueue(
      oclk_abs(N, WrapHandle(a, &ctx), offa, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_abs<float>(const int_tp ctx_id, const int_tp N,
                                      const cl_mem a, const int_tp offa,
                                      cl_mem y, const int_tp offy);
template void greentea_gpu_abs<double>(const int_tp ctx_id, const int_tp N,
                                       const cl_mem a, const int_tp offa,
                                       cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_exp(const int_tp ctx_id, const int_tp N, const cl_mem a,
                      const int_tp offa, cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_exp = program.get_kernel(CL_KERNEL_SELECT("exp"));
  viennacl::ocl::enqueue(
      oclk_exp(N, WrapHandle(a, &ctx), offa, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_exp<float>(const int_tp ctx_id, const int_tp N,
                                      const cl_mem a, const int_tp offa,
                                      cl_mem y, const int_tp offy);
template void greentea_gpu_exp<double>(const int_tp ctx_id, const int_tp N,
                                       const cl_mem a, const int_tp offa,
                                       cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_sqrt(const int_tp ctx_id, const int_tp n,
                       const cl_mem a, const int_tp offa,
                       cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_sqrt = program.get_kernel(
      CL_KERNEL_SELECT("sqrt"));
  viennacl::ocl::enqueue(
      oclk_sqrt(n, WrapHandle(a, &ctx), offa, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_sqrt<float>(const int_tp ctx_id, const int_tp n,
                                       const cl_mem a, const int_tp offa,
                                       cl_mem y, const int_tp offy);
template void greentea_gpu_sqrt<double>(const int_tp ctx_id, const int_tp n,
                                        const cl_mem a, const int_tp offa,
                                        cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_powx(const int_tp ctx_id, const int_tp N, const cl_mem a,
                       const int_tp offa, const Dtype alpha, cl_mem y,
                       const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_powx = program.get_kernel(
      CL_KERNEL_SELECT("powx"));
  viennacl::ocl::enqueue(
      oclk_powx(N, WrapHandle(a, &ctx), offa, alpha, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_powx<float>(const int_tp ctx_id, const int_tp N,
                                       const cl_mem a, const int_tp offa,
                                       const float alpha, cl_mem y,
                                       const int_tp offy);
template void greentea_gpu_powx<double>(const int_tp ctx_id, const int_tp N,
                                        const cl_mem a, const int_tp offa,
                                        const double alpha, cl_mem y,
                                        const int_tp offy);

template<typename Dtype>
void greentea_gpu_log(const int_tp ctx_id, const int_tp N, const cl_mem a,
                      const int_tp offa, cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_log = program.get_kernel(CL_KERNEL_SELECT("log"));
  viennacl::ocl::enqueue(
      oclk_log(N, WrapHandle(a, &ctx), offa, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_log<float>(const int_tp ctx_id, const int_tp N,
                                      const cl_mem a, const int_tp offa,
                                      cl_mem y, const int_tp offy);
template void greentea_gpu_log<double>(const int_tp ctx_id, const int_tp N,
                                       const cl_mem a, const int_tp offa,
                                       cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_sign(const int_tp ctx_id, const int_tp n, const cl_mem x,
int_tp offx,
                       cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_sign = program.get_kernel(
      CL_KERNEL_SELECT("sign"));
  viennacl::ocl::enqueue(
      oclk_sign(n, WrapHandle(x, &ctx), offx, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_sign<float>(const int_tp ctx_id, const int_tp n,
                                       const cl_mem x, int_tp offx, cl_mem y,
                                       const int_tp offy);
template void greentea_gpu_sign<double>(const int_tp ctx_id, const int_tp n,
                                        const cl_mem x, int_tp offx, cl_mem y,
                                        const int_tp offy);

template<typename Dtype>
void greentea_gpu_sgnbit(const int_tp ctx_id, const int_tp n, const cl_mem x,
int_tp offx,
                         cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_sgnbit = program.get_kernel(
      CL_KERNEL_SELECT("sgnbit"));
  viennacl::ocl::enqueue(
      oclk_sgnbit(n, WrapHandle(x, &ctx), offx, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_sgnbit<float>(const int_tp ctx_id, const int_tp n,
                                         const cl_mem x, int_tp offx, cl_mem y,
                                         const int_tp offy);
template void greentea_gpu_sgnbit<double>(const int_tp ctx_id, const int_tp n,
                                          const cl_mem x, int_tp offx, cl_mem y,
                                          const int_tp offy);

void greentea_gpu_rng_uniform(const int_tp ctx_id, const int_tp n, cl_mem r,
int_tp offr) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  std::vector<uint_tp> random(n);  //NOLINT
  caffe_rng_uniform(n, &random[0]);
  greentea_gpu_memcpy(sizeof(uint_tp) * n, &random[0], r, offr, &ctx);
}

template<typename Dtype>
void greentea_gpu_rng_uniform(const int_tp ctx_id, const int_tp n,
                              const Dtype a, const Dtype b, cl_mem r,
                              const int_tp offr) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  std::vector<Dtype> random(n);  // NOLINT
  caffe_rng_uniform(n, a, b, &random[0]);
  greentea_gpu_memcpy(sizeof(Dtype) * n, &random[0], r, offr, &ctx);
}

template void greentea_gpu_rng_uniform<float>(const int_tp ctx_id,
                                              const int_tp n, const float a,
                                              const float b, cl_mem r,
                                              const int_tp offr);
template void greentea_gpu_rng_uniform<double>(const int_tp ctx_id,
                                               const int_tp n, const double a,
                                               const double b, cl_mem r,
                                               const int_tp offr);

template<typename Dtype>
void greentea_gpu_rng_gaussian(const int_tp ctx_id, const int_tp n,
                               const Dtype mu, const Dtype sigma, cl_mem r,
                               const int_tp offr) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  std::vector<Dtype> random(n);  // NOLINT
  caffe_rng_gaussian(n, mu, sigma, &random[0]);
  greentea_gpu_memcpy(sizeof(Dtype) * n, &random[0], r, offr, &ctx);
}

template void greentea_gpu_rng_gaussian<float>(const int_tp ctx_id,
                                               const int_tp n, const float mu,
                                               const float sigma, cl_mem r,
                                               const int_tp offr);

template void greentea_gpu_rng_gaussian<double>(const int_tp ctx_id,
                                                const int_tp n, const double mu,
                                                const double sigma, cl_mem r,
                                                const int_tp offr);

}  // namespace caffe
#endif  // USE_GREENTEA
