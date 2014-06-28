// Copyright 2014 BVLC and contributors.

#ifdef USE_OPENCL
#include "caffe/common.hpp"
#include "caffe/util/opencl_device.hpp"
#include "caffe/util/opencl_math_functions.hpp"

#include <vector>

namespace caffe {

shared_ptr<CaffeOpenCL> CaffeOpenCL::singleton_;

cl_device_type CaffeOpenCL::get_device_type() {
  switch (Caffe::mode()) {
  case Caffe::OPENCL_CPU:
    return CL_DEVICE_TYPE_CPU;
  case Caffe::OPENCL_GPU:
    return CL_DEVICE_TYPE_GPU;
  case Caffe::OPENCL_ALL:
  default:
    return CL_DEVICE_TYPE_ALL;
  }
}

/**
 * http://dhruba.name/2012/08/14/opencl-cookbook-listing-all-devices-and-their-critical-attributes/
 */
void CaffeOpenCL::create_context() {
  cl_uint platformCount;
  CL_CHECK(clGetPlatformIDs(0, NULL, &platformCount));

  cl_platform_id* platforms = (cl_platform_id*)
      malloc(sizeof(cl_platform_id) * platformCount);
  CL_CHECK(clGetPlatformIDs(1, platforms, NULL));

  cl_uint device_count;
  cl_device_type device_type = get_device_type();
  int num_devices_to_skip = current_device_id_;
  while (num_devices_to_skip >= 0) {
    for (int i = 0; i < platformCount; i++) {
      cl_context_properties properties[] = {
          CL_CONTEXT_PLATFORM, (cl_context_properties)(
              platforms[i]), 0};
      // get all devices
      clGetDeviceIDs(platforms[i], device_type, 0, NULL, &device_count);
      if (num_devices_to_skip <= device_count) {
        current_cl_platform_id_ = platforms[i];
        current_platform_device_count_ = device_count;
        current_platform_device_id_ = num_devices_to_skip;
        current_platform_device_ids_.resize(device_count);
        CL_CHECK(clGetDeviceIDs(current_cl_platform_id_, device_type,
                                current_platform_device_count_,
                                &(current_platform_device_ids_[0]), NULL));
        cl_int error = CL_SUCCESS;   // Used to handle error codes
/*
 * http://dhruba.name/2012/10/14/opencl-cookbook-how-to-leverage-multiple-devices-in-opencl/
 * https://software.intel.com/sites/products/documentation/ioclsdk/2013/OG/Using_Shared_Context_for_Multiple_OpenCL_Devices.htm
 */
        cl_context_ = clCreateContext(
            properties, device_count, &(current_platform_device_ids_[0]),
            NULL, NULL, &error);
        CL_CHECK(error);
      }
      num_devices_to_skip -= device_count;
      if (num_devices_to_skip < 0) {
        break;
      }
    }
  }
}

cl_device_id CaffeOpenCL::current_cl_device_id() {
  // To initialize current platform info
  context();
  return current_platform_device_ids_[current_platform_device_id_];
}

void CaffeOpenCL::create_queue() {
  cl_int error = 0;   // Used to handle error codes
  cl_command_queue_properties properties = 0;
  cl_command_queue_ = clCreateCommandQueue(
      context(), current_cl_device_id(), properties, &error);
  CL_CHECK(error);
}

void CaffeOpenCL::release_context() {
  CL_CHECK(clReleaseContext(cl_context_));
  cl_context_ = NULL;
}

void CaffeOpenCL::release_queue() {
  CL_CHECK(clReleaseCommandQueue(cl_command_queue_));
  cl_command_queue_ = NULL;
}

void CaffeOpenCL::SetDevice(const int device_id) {
  if (current_device_id_ != device_id) {
    current_device_id_ = device_id;
    release_queue();
    // TODO: reuse context for the devices of the same platform
    release_context();
    context();
    finalize_clblas();
    initialize_clblas();
  }
}

void CaffeOpenCL::initialize_clblas() {
  if (!clblas_initialized_) {
    CLBLAS_CHECK(clblasSetup());
    clblas_initialized_ = true;
  }
}

void CaffeOpenCL::finalize_clblas() {
  if (clblas_initialized_) {
    clblasTeardown();
    clblas_initialized_ = false;
  }
}

template<typename Dtype>
void OpenCLDevice<Dtype>::gemm(const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const int M,
                        const int N, const int K, const Dtype alpha,
                        const Dtype* A, const Dtype* B,
                        const Dtype beta, Dtype* C) {
  caffe_opencl_gemm(TransA, TransB, M, N, K, alpha, A, B, beta, C);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::gemv(const CBLAS_TRANSPOSE TransA, const int M,
                        const int N, const Dtype alpha, const Dtype* A,
                        const Dtype* x, const Dtype beta, Dtype* y) {
  caffe_opencl_gemv(TransA, M, N, alpha, A, x, beta, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::axpy(const int N, const Dtype alpha,
                        const Dtype* X, Dtype* Y) {
  caffe_opencl_axpy(N, alpha, X, Y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::scal(const int N, const Dtype alpha, Dtype *X) {
  caffe_opencl_scal(N, alpha, X);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::axpby(
    const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y) {
  caffe_opencl_axpby(N, alpha, X, beta, Y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::copy(const int N, const Dtype *X, Dtype *Y) {
  caffe_opencl_copy(N, X, Y);
}


/**
 * http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clEnqueueWriteBuffer.html
 */
template<typename Dtype>
void OpenCLDevice<Dtype>::copy_from_cpu(const int N, const Dtype *X,
                                        Dtype *Y) {
  caffe_opencl_copy_from_cpu(N, X, Y);
}

/**
 * http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clEnqueueFillBuffer.html
 */
template<typename Dtype>
void OpenCLDevice<Dtype>::set(const int N, const Dtype alpha, Dtype *X) {
  caffe_opencl_set(N, alpha, X);
}


//template<typename Dtype>
//void OpenCLDevice<Dtype>::add_scalar(const int N, const Dtype alpha,
//                                     Dtype *X) {
//  NOT_IMPLEMENTED;
//}

//template<typename Dtype>
//void OpenCLDevice<Dtype>::powx(const int N, const Dtype* a, const Dtype b,
//                               Dtype* y) {
//  NOT_IMPLEMENTED;
////  caffe_gpu_powx<Dtype>(N, a, b, y);
//}

//template<typename Dtype>
//void OpenCLDevice<Dtype>::rng_uniform(const int N, const Dtype a,
//                                      const Dtype b, Dtype* r) {
//  NOT_IMPLEMENTED;
////  caffe_gpu_rng_uniform<Dtype>(N, a, b, r);
//}

//template<typename Dtype>
//void OpenCLDevice<Dtype>::rng_gaussian(const int N, const Dtype mu,
//                                       const Dtype sigma, Dtype* r) {
//  NOT_IMPLEMENTED;
////  caffe_gpu_rng_gaussian<Dtype>(N, mu, sigma, r);
//}

//template<typename Dtype>
//void OpenCLDevice<Dtype>::rng_bernoulli(const int N, const Dtype p, int* r) {
//  NOT_IMPLEMENTED;
////  caffe_gpu_rng_bernoulli<Dtype>(N, p, r);
//}

//template<typename Dtype>
//void OpenCLDevice<Dtype>::dot(const int N, const Dtype* x, const Dtype* y,
//                              Dtype* out) {
//  NOT_IMPLEMENTED;
////  caffe_gpu_dot<Dtype>(N, x, y, out);
//}

//template<typename Dtype>
//void OpenCLDevice<Dtype>::hamming_distance(const int N, const Dtype* x,
//                                           const Dtype* y, uint32_t* out) {
//  NOT_IMPLEMENTED;
////  *out = caffe_gpu_hamming_distance<Dtype>(N, x, y);
//}

/**
 *
clblasSasum(
    size_t N,
    cl_mem asum,
    size_t offAsum,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
 */
//template<typename Dtype>
//void OpenCLDevice<Dtype>::asum(const int N, const Dtype* x, Dtype* y) {
//  NOT_IMPLEMENTED;
////  CREATE_CL_MEM(x, N, 1, READ_ONLY);
////  CREATE_CL_MEM(y, N, 1, READ_WRITE);
////  PRE_CLBLAS_CALL;
////  CLBLAS_CHECK(clblasSasum(
////      N, alpha, ARRAY(X),
////      CLBALS_TRAILING_ARGS));
//}

//template<typename Dtype>
//void OpenCLDevice<Dtype>::scale(const int N, const Dtype alpha,
//                                const Dtype *x, Dtype* y) {
//  this->copy(N, x, y);
//  this->scal(N, alpha, y);
//}

//template<typename Dtype>
//void OpenCLDevice<Dtype>::im2col(
//    const Dtype* data_im, const int channels,
//    const int height, const int width, const int ksize, const int pad,
//    const int stride, Dtype* data_col) {
////  NOT_IMPLEMENTED;
////  im2col_gpu(data_im, channels, height, width, ksize, pad, stride,
////             data_col);
//}

//template<typename Dtype>
//void OpenCLDevice<Dtype>::col2im(
//    const Dtype* data_col, const int channels,
//    const int height, const int width, const int psize, const int pad,
//    const int stride, Dtype* data_im) {
////  NOT_IMPLEMENTED;
////  col2im_gpu(data_col, channels, height, width, psize, pad, stride,
////             data_im);
//}


INSTANTIATE_CLASS(OpenCLDevice);

const char* clGetErrorString(cl_int error) {
  switch (error) {
  case CL_SUCCESS:
    return "CL_SUCCESS";
  case CL_INVALID_VALUE:
    return "CL_INVALID_VALUE";
  case CL_INVALID_COMMAND_QUEUE:
    return "CL_INVALID_COMMAND_QUEUE";
  case CL_INVALID_CONTEXT:
    return "CL_INVALID_CONTEXT";
  case CL_INVALID_MEM_OBJECT:
    return "CL_INVALID_MEM_OBJECT";
  case CL_INVALID_DEVICE:
    return "CL_INVALID_DEVICE";
  case CL_INVALID_EVENT_WAIT_LIST:
    return "CL_INVALID_EVENT_WAIT_LIST";
  case CL_OUT_OF_RESOURCES:
    return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY:
    return "CL_OUT_OF_HOST_MEMORY";
  case CL_INVALID_OPERATION:
    return "CL_INVALID_OPERATION";
  case CL_COMPILER_NOT_AVAILABLE:
    return "CL_COMPILER_NOT_AVAILABLE";
  case CL_BUILD_PROGRAM_FAILURE:
    return "CL_BUILD_PROGRAM_FAILURE";
  }
  return "Unknown OpenCL error";
}

const char* clblasGetErrorString(clblasStatus status) {
  switch (status) {
  case clblasSuccess:
    return "clblasSuccess";
  case clblasInvalidValue:
    return "clblasInvalidValue";
  case clblasInvalidCommandQueue:
    return "clblasInvalidCommandQueue";
  case clblasInvalidContext:
    return "clblasInvalidContext";
  case clblasInvalidMemObject:
    return "clblasInvalidMemObject";
  case clblasInvalidDevice:
    return "clblasInvalidDevice";
  case clblasInvalidEventWaitList:
    return "clblasInvalidEventWaitList";
  case clblasOutOfResources:
    return "clblasOutOfResources";
  case clblasOutOfHostMemory:
    return "clblasOutOfHostMemory";
  case clblasInvalidOperation:
    return "clblasInvalidOperation";
  case clblasCompilerNotAvailable:
    return "clblasCompilerNotAvailable";
  case clblasBuildProgramFailure:
    return "clblasBuildProgramFailure";
  case clblasNotImplemented:
    return "clblasNotImplemented";
  case clblasNotInitialized:
    return "clblasNotInitialized";
  case clblasInvalidMatA:
    return "clblasInvalidMatA";
  case clblasInvalidMatB:
    return "clblasInvalidMatB";
  case clblasInvalidMatC:
    return "clblasInvalidMatC";
  case clblasInvalidVecX:
    return "clblasInvalidVecX";
  case clblasInvalidVecY:
    return "clblasInvalidVecY";
  case clblasInvalidDim:
    return "clblasInvalidDim";
  case clblasInvalidLeadDimA:
    return "clblasInvalidLeadDimA";
  case clblasInvalidLeadDimB:
    return "clblasInvalidLeadDimB";
  case clblasInvalidLeadDimC:
    return "clblasInvalidLeadDimC";
  case clblasInvalidIncX:
    return "clblasInvalidIncX";
  case clblasInvalidIncY:
    return "clblasInvalidIncY";
  case clblasInsufficientMemMatA:
    return "clblasInsufficientMemMatA";
  case clblasInsufficientMemMatB:
    return "clblasInsufficientMemMatB";
  case clblasInsufficientMemMatC:
    return "clblasInsufficientMemMatC";
  case clblasInsufficientMemVecX:
    return "clblasInsufficientMemVecX";
  case clblasInsufficientMemVecY:
    return "clblasInsufficientMemVecY";
  }
  return "Unknown clblas status";
}

}  // namespace caffe

#endif  // USE_OPENCL
