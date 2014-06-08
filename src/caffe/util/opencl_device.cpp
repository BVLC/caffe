// Copyright 2014 BVLC and contributors.

#include "caffe/common.hpp"
#include "caffe/util/opencl_device.hpp"

namespace caffe {

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

const char* clblasGetErrorString(clblasStatus_t status) {
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
