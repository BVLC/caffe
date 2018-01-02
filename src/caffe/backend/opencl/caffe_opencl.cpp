#include <string>
#include "caffe/common.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"

namespace caffe {

#ifdef USE_OPENCL

const char* clGetErrorString(cl_int error) {
  switch (error) {
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11: return "CL_BUILD_PROGRAM_FAILURE";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
  case -69: return "CL_INVALID_PIPE_SIZE";
  case -70: return "CL_INVALID_DEVICE_QUEUE";
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  case -1024: return "clBLAS: Functionality is not implemented";
  case -1023: return "clBLAS: Library is not initialized yet";
  case -1022: return "clBLAS: Matrix a is not a valid memory object";
  case -1021: return "clBLAS: Matrix b is not a valid memory object";
  case -1020: return "clBLAS: Matrix c is not a valid memory object";
  case -1019: return "clBLAS: Vector x is not a valid memory object";
  case -1018: return "clBLAS: Vector y is not a valid memory object";
  case -1017: return "clBLAS: An input dimension (m:n:k) is invalid";
  case -1016: return "clBLAS: Leading dimension a must not be less than the "
      "size of the first dimension";
  case -1015: return "clBLAS: Leading dimension b must not be less than the "
      "size of the second dimension";
  case -1014: return "clBLAS: Leading dimension c must not be less than the "
      "size of the third dimension";
  case -1013: return "clBLAS: The increment for a vector x must not be 0";
  case -1012: return "clBLAS: The increment for a vector y must not be 0";
  case -1011: return "clBLAS: The memory object for Matrix a is too small";
  case -1010: return "clBLAS: The memory object for Matrix b is too small";
  case -1009: return "clBLAS: The memory object for Matrix c is too small";
  case -1008: return "clBLAS: The memory object for Vector x is too small";
  case -1007: return "clBLAS: The memory object for Vector y is too small";
  default: return "Unknown OpenCL error";
  }
}

#ifdef USE_FFT
const char* clfftGetErrorString(clfftStatus status) {
  switch (status) {
  case CLFFT_SUCCESS:
    return "CLFFT_SUCCESS";
  case CLFFT_INVALID_PLAN:
    return "CLFFT_INVALID_PLAN";
  case CLFFT_INVALID_GLOBAL_WORK_SIZE:
    return "CLFFT_INVALID_GLOBAL_WORK_SIZE";
  case CLFFT_INVALID_MIP_LEVEL:
    return "CLFFT_INVALID_MIP_LEVEL";
  case CLFFT_INVALID_BUFFER_SIZE:
    return "CLFFT_INVALID_BUFFER_SIZE";
  case CLFFT_INVALID_GL_OBJECT:
    return "CLFFT_INVALID_GL_OBJECT";
  case CLFFT_INVALID_OPERATION:
    return "CLFFT_INVALID_OPERATION";
  case CLFFT_INVALID_EVENT:
    return "CLFFT_INVALID_EVENT";
  case CLFFT_INVALID_EVENT_WAIT_LIST:
    return "CLFFT_INVALID_EVENT_WAIT_LIST";
  case CLFFT_INVALID_GLOBAL_OFFSET:
    return "CLFFT_INVALID_GLOBAL_OFFSET";
  case CLFFT_INVALID_WORK_ITEM_SIZE:
    return "CLFFT_INVALID_WORK_ITEM_SIZE";
  case CLFFT_INVALID_WORK_GROUP_SIZE:
    return "CLFFT_INVALID_WORK_GROUP_SIZE";
  case CLFFT_INVALID_WORK_DIMENSION:
    return "CLFFT_INVALID_WORK_DIMENSION";
  case CLFFT_INVALID_KERNEL_ARGS:
    return "CLFFT_INVALID_KERNEL_ARGS";
  case CLFFT_INVALID_ARG_SIZE:
    return "CLFFT_INVALID_ARG_SIZE";
  case CLFFT_INVALID_ARG_VALUE:
    return "CLFFT_INVALID_ARG_VALUE";
  case CLFFT_INVALID_ARG_INDEX:
    return "CLFFT_INVALID_ARG_INDEX";
  case CLFFT_INVALID_KERNEL:
    return "CLFFT_INVALID_KERNEL";
  case CLFFT_INVALID_KERNEL_DEFINITION:
    return "CLFFT_INVALID_KERNEL_DEFINITION";
  case CLFFT_INVALID_KERNEL_NAME:
    return "CLFFT_INVALID_KERNEL_NAME";
  case CLFFT_INVALID_PROGRAM_EXECUTABLE:
    return "CLFFT_INVALID_PROGRAM_EXECUTABLE";
  case CLFFT_INVALID_PROGRAM:
    return "CLFFT_INVALID_PROGRAM";
  case CLFFT_INVALID_BUILD_OPTIONS:
    return "CLFFT_INVALID_BUILD_OPTIONS";
  case CLFFT_INVALID_BINARY:
    return "CLFFT_INVALID_BINARY";
  case CLFFT_INVALID_SAMPLER:
    return "CLFFT_INVALID_SAMPLER";
  case CLFFT_INVALID_IMAGE_SIZE:
    return "CLFFT_INVALID_IMAGE_SIZE";
  case CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case CLFFT_INVALID_MEM_OBJECT:
    return "CLFFT_INVALID_MEM_OBJECT";
  case CLFFT_INVALID_HOST_PTR:
    return "CLFFT_INVALID_HOST_PTR";
  case CLFFT_INVALID_COMMAND_QUEUE:
    return "CLFFT_INVALID_COMMAND_QUEUE";
  case CLFFT_INVALID_QUEUE_PROPERTIES:
    return "CLFFT_INVALID_QUEUE_PROPERTIES";
  case CLFFT_INVALID_CONTEXT:
    return "CLFFT_INVALID_CONTEXT";
  case CLFFT_INVALID_DEVICE:
    return "CLFFT_INVALID_DEVICE";
  case CLFFT_INVALID_PLATFORM:
    return "CLFFT_INVALID_PLATFORM";
  case CLFFT_INVALID_DEVICE_TYPE:
    return "CLFFT_INVALID_DEVICE_TYPE";
  case CLFFT_INVALID_VALUE:
    return "CLFFT_INVALID_VALUE";
  case CLFFT_MAP_FAILURE:
    return "CLFFT_MAP_FAILURE";
  case CLFFT_BUILD_PROGRAM_FAILURE:
    return "CLFFT_BUILD_PROGRAM_FAILURE";
  case CLFFT_IMAGE_FORMAT_NOT_SUPPORTED:
    return "CLFFT_IMAGE_FORMAT_NOT_SUPPORTED";
  case CLFFT_IMAGE_FORMAT_MISMATCH:
    return "CLFFT_IMAGE_FORMAT_MISMATCH";
  case CLFFT_MEM_COPY_OVERLAP:
    return "CLFFT_MEM_COPY_OVERLAP";
  case CLFFT_PROFILING_INFO_NOT_AVAILABLE:
    return "CLFFT_PROFILING_INFO_NOT_AVAILABLE";
  case CLFFT_OUT_OF_HOST_MEMORY:
    return "CLFFT_OUT_OF_HOST_MEMORY";
  case CLFFT_OUT_OF_RESOURCES:
    return "CLFFT_OUT_OF_RESOURCES";
  case CLFFT_MEM_OBJECT_ALLOCATION_FAILURE:
    return "CLFFT_MEM_OBJECT_ALLOCATION_FAILURE";
  case CLFFT_COMPILER_NOT_AVAILABLE:
    return "CLFFT_COMPILER_NOT_AVAILABLE";
  case CLFFT_DEVICE_NOT_AVAILABLE:
    return "CLFFT_DEVICE_NOT_AVAILABLE";
  case CLFFT_DEVICE_NOT_FOUND:
    return "CLFFT_DEVICE_NOT_FOUND";
  case CLFFT_BUGCHECK:
    return "CLFFT_BUGCHECK";
  case CLFFT_NOTIMPLEMENTED:
    return "CLFFT_NOTIMPLEMENTED";
  case CLFFT_TRANSPOSED_NOTIMPLEMENTED:
    return "CLFFT_TRANSPOSED_NOTIMPLEMENTED";
  case CLFFT_FILE_NOT_FOUND:
    return "CLFFT_FILE_NOT_FOUND";
  case CLFFT_FILE_CREATE_FAILURE:
    return "CLFFT_FILE_CREATE_FAILURE";
  case CLFFT_VERSION_MISMATCH:
    return "CLFFT_VERSION_MISMATCH";
  case CLFFT_DEVICE_NO_DOUBLE:
    return "CLFFT_DEVICE_NO_DOUBLE";
  case CLFFT_DEVICE_MISMATCH:
    return "CLFFT_DEVICE_MISMATCH";
  default:
    return "CLFFT_UNKNOWN_ERROR";
  }
}

#endif  // USE FFT

#endif  // USE_OPENCL


}  // namespace caffe
