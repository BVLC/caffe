#ifdef USE_OPENCL

#include <caffe/util/OpenCL/OpenCLSupport.hpp>

#include <CL/cl.h>
#include <clBLAS.h>
#include <glog/logging.h>
#include <limits.h>
#include <tr1/memory>

#include <caffe/syncedmem.hpp>
#include <caffe/util/benchmark.hpp>
#include <caffe/util/OpenCL/definitions.hpp>

#include <cstdlib>
#include <exception>
#include <map>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace caffe {

namespace OpenCL {

const char* what(cl_int value) {
  int errorCode = static_cast<int>(value);
  switch (errorCode) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
      return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
      return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
      return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
      return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
      return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
      return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_QUEUED:
      return "CL_QUEUED";
    case CL_SUBMITTED:
      return "CL_SUBMITTED";
    case CL_RUNNING:
      return "CL_RUNNING";
    case clblasNotImplemented:
      return "clBLAS: Functionality is not implemented";
    case clblasNotInitialized:
      return "clBLAS library is not initialized yet";
    case clblasInvalidMatA:
      return "clBLAS Matrix A is not a valid memory object";
    case clblasInvalidMatB:
      return "clBLAS Matrix B is not a valid memory object";
    case clblasInvalidMatC:
      return "clBLAS Matrix C is not a valid memory object";
    case clblasInvalidVecX:
      return "clBLAS Vector X is not a valid memory object";
    case clblasInvalidVecY:
      return "clBLAS Vector Y is not a valid memory object";
    case clblasInvalidDim:
      return "clBLAS An input dimension (M, N, K) is invalid";
    case clblasInvalidLeadDimA:
      return "clBLAS Leading dimension A "
              "must not be less than the size of the first dimension";
    case clblasInvalidLeadDimB:
      return "clBLAS Leading dimension B "
              "must not be less than the size of the second dimension";
    case clblasInvalidLeadDimC:
      return "clBLAS Leading dimension C "
              "must not be less than the size of the third dimension";
    case clblasInvalidIncX:
      return "clBLAS The increment for a vector X must not be 0";
    case clblasInvalidIncY:
      return "clBLAS The increment for a vector Y must not be 0";
    case clblasInsufficientMemMatA:
      return "clBLAS The memory object for Matrix A is too small";
    case clblasInsufficientMemMatB:
      return "clBLAS The memory object for Matrix B is too small";
    case clblasInsufficientMemMatC:
      return "clBLAS The memory object for Matrix C is too small";
    case clblasInsufficientMemVecX:
      return "clBLAS The memory object for Vector X is too small";
    case clblasInsufficientMemVecY:
      return "clBLAS The memory object for Vector Y is too small";
      /* FIXME: check what platform this addresses
       case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR:
       return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
       case CL_PLATFORM_NOT_FOUND_KHR:
       return "CL_PLATFORM_NOT_FOUND_KHR";
       case CL_INVALID_PROPERTY_EXT:
       return "CL_INVALID_PROPERTY_EXT";
       case CL_DEVICE_PARTITION_FAILED_EXT:
       return "CL_DEVICE_PARTITION_FAILED_EXT";
       case CL_INVALID_PARTITION_COUNT_EXT:
       return "CL_INVALID_PARTITION_COUNT_EXT";
       */
    default:
      return "unknown error code";
  }
  return "unknown error code";
}

template<typename T>
std::string clGetKernelName(std::string name) {
  std::map<std::pair<std::string, std::type_index>, std::string>::iterator it;
  it = caffe::OpenCL::mapKernelName.find(
      std::make_pair(name, std::type_index(typeid(T))));
  if (it != mapKernelName.end()) {
    return it->second;
  }

  std::stringstream ss;
  ss << name;


  float f;
  double d;
  char c;
  int i;

  const type_info& floatID  = typeid(f);
  const type_info& doubleID  = typeid(d);
  const type_info& charID    = typeid(c);
  const type_info& intID     = typeid(i);

  if (typeid(T) == floatID) {
    ss << "Float";
    caffe::OpenCL::mapKernelName[std::make_pair(name,
        std::type_index(floatID))] = ss.str();
    return ss.str();
  }
  if (typeid(T) == doubleID) {
    ss << "Double";
    caffe::OpenCL::mapKernelName[std::make_pair(name,
        std::type_index(doubleID))] = ss.str();
    return ss.str();
  }
  if (typeid(T) == charID) {
    ss << "Char";
    caffe::OpenCL::mapKernelName[std::make_pair(name,
        std::type_index(charID))] = ss.str();
    return ss.str();
  }
  if (typeid(T) == intID) {
    ss << "Int";
    caffe::OpenCL::mapKernelName[std::make_pair(name,
        std::type_index(intID))] = ss.str();
    return ss.str();
  }

  return ss.str();
}
template std::string clGetKernelName<double>(std::string name);
template std::string clGetKernelName<float>(std::string name);

bool clMalloc(void** virtualPtr, size_t size) {
  std::tr1::shared_ptr<OpenCLPlatform> pf = OpenCLManager::CurrentPlatform();
  OpenCLDevice& device = pf->CurrentDevice();

  OpenCLMemory clMem;
  try {
    clMem = OpenCLMemory(size);
  } catch (std::exception& e) {
    return false;
  }
  device.add(clMem);

  *virtualPtr = clMem.getVirtualPointer();
  return true;
}

bool clGetBuffer(void** virtualPtr, size_t size) {
  std::tr1::shared_ptr<OpenCLPlatform> pf = OpenCLManager::CurrentPlatform();
  OpenCLDevice& device = pf->CurrentDevice();

  std::tr1::shared_ptr<OpenCLBuffer> buffer =
      std::tr1::shared_ptr<OpenCLBuffer>(new OpenCLBuffer(size));

  if (!device.getBuffer(&buffer, size)) {
    return false;
  }
  buffer->setUnavailable();
  *virtualPtr = buffer->getVirtualPointer();

  return true;
}

bool clBufferSetAvailable(void* virtualPtr, size_t size) {
  std::tr1::shared_ptr<OpenCLPlatform> pf = OpenCLManager::CurrentPlatform();
  OpenCLDevice& device = pf->CurrentDevice();

  return device.setBufferAvailable(virtualPtr, size);
}

bool clFree(void* virtualPtr) {
  std::tr1::shared_ptr<OpenCLPlatform> pf = OpenCLManager::CurrentPlatform();
  OpenCLDevice& device = pf->CurrentDevice();

  if (!device.isValidPtr(virtualPtr)) {
    LOG(ERROR) << device.name() << "> not a valid memory pointer @ "
    << virtualPtr;
    return false;
  }

  OpenCLMemory* clMem;
  if (!device.get(virtualPtr, &clMem)) {
    LOG(ERROR) << device.name()
    << "> failed to get OpenCLMemory object using virtual pointer @ "
    << virtualPtr;
    return false;
  }

  if (clMem->hasEvent()) {
    cl_event event = clMem->getEvent();
    DLOG(INFO) << "waiting for event of " << clMem->getTag();
    CL_CHECK(clWaitForEvents(1, &event));
  }
  device.rmMemoryPtr(clMem->getVirtualPointer());
  // clMem = NULL;
  // device.Synchronize();

  return true;
}

template<typename T>
bool clMemset(void* virtualPtr, const T alpha, const size_t Bytes) {
  std::tr1::shared_ptr<OpenCLPlatform> pf = OpenCLManager::CurrentPlatform();
  OpenCLDevice& device = pf->CurrentDevice();
  cl_command_queue* queue = device.getQueue();

  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  if (!device.isValidPtr(virtualPtr)) {
    LOG(ERROR) << device.name()
    << "> not a valid memory pointer @ "
    << virtualPtr;
    return false;
  }

  OpenCLMemory* clMem;
  if (!device.get(virtualPtr, &clMem)) {
    LOG(ERROR) << device.name()
    << "> failed to get GPU memory @ "
    << virtualPtr;
    return false;
  }

  cl_event bufferEvent = NULL;

#ifdef OPENCL_VERSION_1_2  // at least OpenCL Version 1.2 required, this is slow

  size_t mem_offset = clGetMemoryOffset(virtualPtr);
  cl_mem base = (cl_mem) clMem->getLogicalPointer();

  cl_int err = clEnqueueFillBuffer(*queue, base, &alpha, sizeof(T), mem_offset,
      Bytes, 0, NULL, &bufferEvent);
  // clWaitForEvents(1, &bufferEvent);
  if (err != CL_SUCCESS) {
    std::ostringstream oss;
    oss << "clEnqueueFillBuffer() failed on GPU " << device.name() << " : "
        << what(err);
    // LOG(ERROR) << oss.str();
    throw OpenCLSupportException(oss.str());
    return false;
  }
  DLOG(INFO) << "clEnqueueFillBuffer() succeeded to set memory to " << alpha;

#else

  std::string kernel_name = clGetKernelName<T>("clFillBuffer");

  cl_kernel* kernel = device.getKernel(kernel_name);
  if (kernel == NULL) {
    return false;
  }

  unsigned int N = Bytes/sizeof(T);
  CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, N, kernel)
  CL_SET_TYPE_KERNEL_ARG(T, alpha, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&virtualPtr, kernel)

  size_t global = CAFFE_GET_GLOBAL_WORKITEMS(N, OPENCL_LOCAL_SIZE);
  size_t local = CAFFE_GET_LOCAL_WORKITEMS(N, OPENCL_LOCAL_SIZE);

  err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL,
      &global, &local, 0, NULL, &bufferEvent);

  if ( err != CL_SUCCESS ) {
    std::ostringstream oss;
    oss << "Failed to enqueue kernel '"
    << kernel_name << "' on GPU " << device.name() << " : " << what(err);
    LOG(ERROR) << oss.str();
    throw OpenCLSupportException(oss.str());
    return false;
  }
  // clWaitForEvents(1, &bufferEvent);
  DLOG(INFO) << "kernel '" << kernel_name << "' executed on GPU "
  << device.name();

  CL_SET_KERNEL_ARG_END
#endif

  DLOG(INFO) << device.name() << "> set OpenCL memory at " <<
  device.getMemoryTag(virtualPtr) << " to " << alpha;

  clMem->setEvent(bufferEvent);

  return true;
}
template bool clMemset<char>(void* gpuPtr, const char alpha, const size_t N);
template bool clMemset<int>(void* gpuPtr, const int alpha, const size_t N);
template bool clMemset<float>(void* gpuPtr, const float alpha, const size_t N);
template bool clMemset<double>(
    void* gpuPtr,
    const double alpha,
    const size_t N);

bool clMemcpy(
    void* virtualDstPtr,
    const void* virtualSrcPtr,
    size_t size,
    int type) {
  std::string function = __func__;

  std::tr1::shared_ptr<OpenCLPlatform> pf = OpenCLManager::CurrentPlatform();
  OpenCLDevice& device = pf->CurrentDevice();
  cl_command_queue* queue = device.getQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  OpenCLMemory *clMemSrc, *clMemDst;
  const void *baseSrc;
  const void* baseDst;
  size_t offsetSrc, offsetDst;
  cl_event copyEvent;

  switch (type) {
    case caffe::OpenCL::COPY_CPU_TO_CPU:
      if (device.isValidPtr(virtualDstPtr)) {
        LOG(ERROR) << device.name()
        << "> dst pointer is in GPU memory @ " << virtualDstPtr;
        return false;
      }
      memcpy(virtualDstPtr, virtualSrcPtr, size);  // NOLINT(*)
      DLOG(INFO) << device.name() << "> copy CPU@" << virtualSrcPtr
      << " to CPU@" << virtualDstPtr << " " << size
      << " Byte transferred.";
      break;
      case caffe::OpenCL::COPY_CPU_TO_GPU:
      if (!device.isValidPtr(virtualDstPtr)) {
        LOG(ERROR) << device.name() << "> dst pointer is not in GPU memory @ "
        << virtualDstPtr;
        return false;
      }

      if (!device.get(virtualDstPtr, &clMemDst)) {
        LOG(ERROR) << device.name() << "> failed to get GPU memory @ "
        << virtualDstPtr;
        return false;
      }
      baseDst = clMemDst->getLogicalPointer();
      offsetDst = clGetMemoryOffset(virtualDstPtr);

      if ( !CL_CHECK(clEnqueueWriteBuffer(*queue, (cl_mem) baseDst, CL_TRUE,
                  offsetDst, size, virtualSrcPtr,
                  0, NULL, &copyEvent) ) ) {
        LOG(ERROR) << device.name() << "> copy CPU@" << virtualSrcPtr
        << " to " << device.getMemoryTag(virtualDstPtr).c_str()
        << " " << size << " Byte transferred failed.";
        return false;
      }

      DLOG(INFO) << device.name() << "> copy CPU@" << virtualSrcPtr << " to "
      << device.getMemoryTag(virtualDstPtr) << " " << size
      << " Byte transferred.";

      // clWaitForEvents(1, &copyEvent);
      clMemDst->setEvent(copyEvent);

      break;
      case caffe::OpenCL::COPY_GPU_TO_CPU:
      if (!device.isValidPtr(virtualSrcPtr)) {
        LOG(ERROR) << device.name()
        << "> src pointer is not in GPU memory @ "
        << virtualSrcPtr;
        return false;
      }
      if (!device.get(virtualSrcPtr, &clMemSrc)) {
        LOG(ERROR) << device.name() << "> failed to get GPU memory @ "
        << virtualSrcPtr;
        return false;
      }
      baseSrc = clMemSrc->getLogicalPointer();
      offsetSrc = clGetMemoryOffset(virtualSrcPtr);

      if (!CL_CHECK(clEnqueueReadBuffer(*queue, (cl_mem) baseSrc, CL_TRUE,
                  offsetSrc, size, virtualDstPtr, 0,
                  NULL, &copyEvent) ) ) {
        return false;
      }

      DLOG(INFO) << device.name() << "> copy "
      << device.getMemoryTag(virtualSrcPtr) << " to CPU@"
      << virtualDstPtr << " " << size << " Byte transferred.";

      // clWaitForEvents(1, &copyEvent);
      clMemSrc->setEvent(copyEvent);

      break;

      case caffe::OpenCL::COPY_GPU_TO_GPU:
      if (!device.isValidPtr(virtualSrcPtr)) {
        LOG(ERROR) << device.name()
        << "> src pointer is not in GPU memory @ "
        << virtualSrcPtr;
        return false;
      }
      if (!device.isValidPtr(virtualDstPtr)) {
        LOG(ERROR) << device.name()
        << "> dst pointer is not in GPU memory @ "
        << virtualDstPtr;
        return false;
      }
      DLOG(INFO) << "VMS@" << virtualSrcPtr << " to VMD@"
      << virtualDstPtr << " size = " << size;

      if (!device.get(virtualSrcPtr, &clMemSrc)) {
        LOG(ERROR) << device.name() << "> failed to get GPU memory @ "
        << virtualSrcPtr;
        return false;
      }
      baseSrc = clMemSrc->getLogicalPointer();
      offsetSrc = clGetMemoryOffset(virtualSrcPtr);
      DLOG(INFO) << device.getMemoryTag(clMemSrc->getVirtualPointer()).c_str();
      DLOG(INFO) << "VMS@" << baseSrc << " offset = " << offsetSrc;

      if (!device.get(virtualDstPtr, &clMemDst)) {
        LOG(ERROR) << device.name() << "> failed to get GPU memory @ "
        << virtualDstPtr;
        return false;
      }
      baseDst = clMemDst->getLogicalPointer();
      offsetDst = clGetMemoryOffset(virtualDstPtr);
      DLOG(INFO) << device.getMemoryTag(clMemDst->getVirtualPointer());
      DLOG(INFO) << "VMD@" << baseDst << " offset = " << offsetDst;

      if (offsetSrc + size > clMemSrc->getSize()) {
        LOG(ERROR) << device.name() << "> copy range out of source range.";
        return false;
      }

      if (offsetDst + size > clMemDst->getSize()) {
        LOG(ERROR) << device.name()
        << "> copy range out of destination range.";
        return false;
      }

      if ( clMemSrc->getVirtualPointer() != clMemDst->getVirtualPointer() ) {
        DLOG(INFO) << "no overlap";

        if ( !CL_CHECK(clEnqueueCopyBuffer(*queue, (cl_mem) baseSrc,
                    (cl_mem) baseDst, offsetSrc,
                    offsetDst, size, 0, NULL, &copyEvent) ) ) {
          return false;
        }
        // clWaitForEvents(1, &copyEvent);
        clMemDst->setEvent(copyEvent);
      } else {
        DLOG(INFO) << "caffe::OpenCL::COPY_GPU_TO_GPU: with overlap";

        void* bufVPtr = NULL;
        if (!clMalloc(&bufVPtr, size)) {
          LOG(ERROR) << device.name()
          << "> failed to created buffer of size = "
          << size << " Byte";
          return false;
        }

        OpenCLMemory* buffer;
        if (!device.get(bufVPtr, &buffer)) {
          LOG(ERROR) << device.name()
          << "> failed to get buffer Object GPU@VIRT"
          << bufVPtr;
          return false;
        }
        const void* bufLPtr = buffer->getLogicalPointer();

        if (!CL_CHECK(
            clEnqueueCopyBuffer(
                *queue,
                (cl_mem) baseSrc,
                (cl_mem) bufLPtr,
                offsetSrc,
                0,
                size,
                0,
                NULL,
                &copyEvent))) {
          return false;
        }
        // CL_CHECK(clWaitForEvents(1, &copyEvent) );
        DLOG(INFO) << device.name()
                   << "> copy "
                   << device.getMemoryTag(virtualSrcPtr).c_str()
                   << " with offset "
                   << offsetSrc
                   << " to buffer "
                   << device.getMemoryTag(bufVPtr).c_str()
                   << " "
                   << size
                   << " Byte transferred.";

        if ( !CL_CHECK(
            clEnqueueCopyBuffer(
                *queue,
                (cl_mem) bufLPtr,
                (cl_mem) baseDst,
                0,
                offsetDst,
                size,
                0,
                NULL,
                &copyEvent) ) ) {
          return false;
        }
        // clWaitForEvents(1, &copyEvent);
        clMemDst->setEvent(copyEvent);

        DLOG(INFO) << device.name()
                   << "> copy "
                   << device.getMemoryTag(bufVPtr).c_str()
                   << " to "
                   << device.getMemoryTag(virtualDstPtr).c_str()
                   << " with offset "
                   << offsetDst
                   << " "
                   << size
                   << " Byte transferred.";

        clFree(bufVPtr);
      }

      DLOG(INFO) << device.name()
                 << "> copy "
                 << device.getMemoryTag(virtualSrcPtr).c_str()
                 << " with offset "
                 << offsetSrc
                 << " to "
                 << device.getMemoryTag(virtualDstPtr).c_str()
                 << " with offset "
                 << offsetDst
                 << " "
                 << size
                 << " Byte transferred.";
      break;

      case caffe::OpenCL::COPY_DEFAULT:
      if ( !device.isValidPtr(virtualSrcPtr) &&
           !device.isValidPtr(virtualDstPtr) ) {
        return clMemcpy(
            virtualDstPtr,
            virtualSrcPtr,
            size,
            caffe::OpenCL::COPY_CPU_TO_CPU);
      }
      if ( !device.isValidPtr(virtualSrcPtr) &&
            device.isValidPtr(virtualDstPtr) ) {
        return clMemcpy(
            virtualDstPtr,
            virtualSrcPtr,
            size,
            caffe::OpenCL::COPY_CPU_TO_GPU);
      }
      if ( device.isValidPtr(virtualSrcPtr) &&
          !device.isValidPtr(virtualDstPtr) ) {
        return clMemcpy(
            virtualDstPtr,
            virtualSrcPtr,
            size,
            caffe::OpenCL::COPY_GPU_TO_CPU);
      }
      if ( device.isValidPtr(virtualSrcPtr) &&
           device.isValidPtr(virtualDstPtr) ) {
        return clMemcpy(
            virtualDstPtr,
            virtualSrcPtr,
            size,
            caffe::OpenCL::COPY_GPU_TO_GPU);
      }
      break;

      default:
      LOG(ERROR) << "unsupported copy mode = " << type;
      return false;
      break;
    }

  return true;
}

bool clReleaseSubBuffers(std::vector<cl_mem>& subBuffers) {  // NOLINT(*)
  for (std::vector<cl_mem>::iterator it = subBuffers.begin();
      it != subBuffers.end(); it++) {
    cl_int err = clReleaseMemObject(*it);
    if (err != CL_SUCCESS) {
      LOG(ERROR) << "failed to release sub-buffer";
      return false;
    }
  }
  return true;
}

bool clReleaseBufferMap(
  std::map<const void*, std::pair<void*, size_t> >& bufferMap) {    // NOLINT(*)
  for (std::map<const void*, std::pair<void*, size_t> >::iterator it =
      bufferMap.begin(); it != bufferMap.end(); it++) {
    const void* virtual_ptr = it->first;
    void* buffer_ptr = bufferMap[virtual_ptr].first;
    size_t buffer_size = bufferMap[virtual_ptr].second;
    if (!clMemcpy(const_cast<void*>(virtual_ptr), buffer_ptr, buffer_size,
        caffe::OpenCL::COPY_DEFAULT)) {
      return false;
    }
    clFree(reinterpret_cast<void*>(buffer_ptr));
  }
  return true;
}

bool clSetKernelArrayArg(
    const void* ptr_virtual,
    unsigned int& idx,  // NOLINT(*)
    std::vector<cl_mem>& subBuffers,  // NOLINT(*)
    std::map<const void*, std::pair<void*, size_t> >& bufferMap,  // NOLINT(*)
    cl_kernel* kernel) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  cl_int err;

  /* case when array is NULL */
  if (ptr_virtual == NULL) {
    DLOG(INFO) << "kernel variable pointer is NULL for kernel argument "
    << idx;
    err = clSetKernelArg(*kernel, idx, sizeof(cl_mem), NULL);
    if ( err != CL_SUCCESS ) {
      LOG(ERROR) << "failed to set kernel argument "
      << idx << " for kernel on GPU "
      << device.name() << " : " << what(err);
      return false;
    }
    idx++;
    return true;
  }

  /* get cl_mem location using the virtual memory pointer */
  const void* ptr_logical;
  if (!clMakeLogical(ptr_virtual, &ptr_logical)) {
    return false;
  }

  size_t offset = clGetMemoryOffset(ptr_virtual);
  size_t size = clGetMemorySize(ptr_virtual);

  /* case when there is no offset to base of cl_mem */
  if (offset == 0) {
    err = clSetKernelArg(*kernel, idx, sizeof(ptr_logical), &ptr_logical);
    if (err != CL_SUCCESS) {
      LOG(ERROR) << "failed to set kernel argument "
                 << idx
                 << " for kernel on GPU "
                 << device.name()
                 << " : "
                 << what(err);
      return false;
    }
    idx++;
    return true;
  }

  /* check alignment */
  cl_uint bytes = device.getDeviceMemBaseAddrAlign();
  if (offset % bytes != 0) {
    LOG(WARNING) << "sub-buffer memory offset ("
                 << offset
                 << " Byte) is not aligned with device memory offset ("
                 << bytes
                 << " Byte)";

    cl_mem_flags flags;
    err = clGetMemObjectInfo((cl_mem) ptr_logical, CL_MEM_FLAGS,
                              sizeof(flags), &flags, NULL);
    if ( err != CL_SUCCESS ) {
      LOG(ERROR) << "failed to query memory properties.";
      return false;
    }

    void* ptr_buffer;
    size_t buffer_size = size-offset;
    if ( !clMalloc(&ptr_buffer, buffer_size) ) {
      return false;
    }

    if ( !clMemcpy(ptr_buffer, ptr_virtual,
                   buffer_size, caffe::OpenCL::COPY_DEFAULT) ) {
      return false;
    }

    if ( !clMakeLogical(ptr_buffer, &ptr_logical) ) {
      return false;
    }
    bufferMap[ptr_virtual] = std::make_pair(ptr_buffer, buffer_size);

    err = clSetKernelArg(*kernel, idx, sizeof(ptr_logical), &ptr_logical);
    if ( err != CL_SUCCESS ) {
      LOG(ERROR) << "failed to set kernel argument " << idx
      << " for kernel on GPU " << device.name()
      << " : " << what(err);
      return false;
    }
    idx++;
    return true;
  }

  /* case when there is an offset to base of cl_mem */
  DLOG(INFO) << "memory offset = " << offset
  << " detected, create sub-buffer [" << offset << "|"
  << size-offset << "]";
  cl_buffer_region region = { offset, size - offset };
  cl_mem sb = clCreateSubBuffer((cl_mem) ptr_logical, CL_MEM_READ_WRITE,
  CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "failed to create sub-buffer for kernel argument "
    << idx << " for kernel on GPU "
    << device.name() << " : " << what(err);
    return false;
  }
  DLOG(INFO) << "create sub-buffer " << subBuffers.size() << std::endl;
  subBuffers.push_back(sb);

  err = clSetKernelArg(*kernel, idx, sizeof(sb), &sb);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "failed to set kernel argument " << idx
    << " for kernel on GPU "
    << device.name() << " : " << what(err);
    return false;
  }
  idx++;

  return true;
}

template<typename T>
bool clSetKernelTypeArg(
    T variable, unsigned int& idx, cl_kernel* kernel) {  // NOLINT(*)
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  cl_int err;

  err = clSetKernelArg(*kernel, idx, sizeof(T), &variable);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "failed to set kernel argument "
               << idx
               << " for kernel on GPU "
               << device.name()
               << " : "
               << what(err);
    return false;
  }
  idx++;
  return true;
}
template bool clSetKernelTypeArg<int>(
    int variable,
    unsigned int& idx,  // NOLINT(*)
    cl_kernel* kernel);
template bool clSetKernelTypeArg<const int>(
    const int variable,
    unsigned int& idx,  // NOLINT(*)
    cl_kernel* kernel);
template bool clSetKernelTypeArg<unsigned int>(
    unsigned int variable,
    unsigned int& idx,  // NOLINT(*)
    cl_kernel* kernel);
template bool clSetKernelTypeArg<const unsigned int>(
    const unsigned int variable,
    unsigned int& idx,  // NOLINT(*)
    cl_kernel* kernel);
template bool clSetKernelTypeArg<float>(
    float variable,
    unsigned int& idx,  // NOLINT(*)
    cl_kernel* kernel);
template bool clSetKernelTypeArg<const float>(
    const float variable,
    unsigned int& idx,  // NOLINT(*)
    cl_kernel* kernel);
template bool clSetKernelTypeArg<double>(
    double variable,
    unsigned int& idx,  // NOLINT(*)
    cl_kernel* kernel);
template bool clSetKernelTypeArg<const double>(
    const double variable,
    unsigned int& idx,  // NOLINT(*)
    cl_kernel* kernel);

template<typename T>
bool clBLASasum(const int N, const void* array_virtual, T* y) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  void* asum;
  caffe::OpenCL::clGetBuffer(&asum, sizeof(T));
  const void* asum_device;
  if (!clMakeLogical(asum, &asum_device)) {
    caffe::OpenCL::clBufferSetAvailable(asum, sizeof(T));
    return false;
  }

  void* buf;
  caffe::OpenCL::clGetBuffer(&buf, N * sizeof(T));
  const void* buf_device;
  if (!clMakeLogical(buf, &buf_device)) {
    caffe::OpenCL::clBufferSetAvailable(buf, N * sizeof(T));
    return false;
  }

  /*
   caffe::SyncedMemory asum(sizeof(T));
   const void* asum_virtual = asum.mutable_gpu_data();
   const void* asum_logical;
   if ( !clMakeLogical(asum_virtual, &asum_logical) ) {
   return false;
   }

   caffe::SyncedMemory buf(N*sizeof(T));
   const void* buf_virtual  = buf.mutable_gpu_data();
   const void* buf_logical;
   if ( !clMakeLogical(buf_virtual, &buf_logical) ) {
   return false;
   }
   */
  const void* array_logical;
  if (!clMakeLogical(array_virtual, &array_logical)) {
    caffe::OpenCL::clBufferSetAvailable(asum, sizeof(T));
    caffe::OpenCL::clBufferSetAvailable(buf, N * sizeof(T));
    return false;
  }

  float f;
  double d;
  const type_info& floatID   = typeid(f);
  const type_info& doubleID  = typeid(d);

  if (typeid(T) == floatID) {
    if (!CL_CHECK(clblasSasum(N,
                              (cl_mem) asum_device, 0,
                              (cl_mem) array_logical, 0, 1,
                              (cl_mem) buf_device, 1, queue, 0, NULL, NULL))) {
      caffe::OpenCL::clBufferSetAvailable(asum, sizeof(T));
      caffe::OpenCL::clBufferSetAvailable(buf, N * sizeof(T));
      return false;
    }
  }

  if (typeid(T) == doubleID) {
    if (!CL_CHECK(clblasDasum(N,
                              (cl_mem) asum_device, 0,
                              (cl_mem) array_logical, 0, 1,
                              (cl_mem) buf_device, 1, queue, 0, NULL, NULL))) {
      caffe::OpenCL::clBufferSetAvailable(asum, sizeof(T));
      caffe::OpenCL::clBufferSetAvailable(buf, N * sizeof(T));
      return false;
    }
  }

  // *y = *((T*) asum.mutable_cpu_data());
  caffe::OpenCL::clMemcpy(y, asum, sizeof(T), caffe::OpenCL::COPY_GPU_TO_CPU);
  caffe::OpenCL::clBufferSetAvailable(asum, sizeof(T));
  caffe::OpenCL::clBufferSetAvailable(buf, N * sizeof(T));

  return true;
}
template bool clBLASasum<float>(
    const int N,
    const void* array_GPU_ptr,
    float* y);
template bool clBLASasum<double>(
    const int N,
    const void* array_GPU_ptr,
    double* y);

template<typename T>
bool clsign(const int n, const void* array_GPU_x, void* array_GPU_y) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  std::string kernel_name = clGetKernelName<T>("clsign");

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  cl_kernel* kernel = device.getKernel(kernel_name);
  if (kernel == NULL) {
    return false;
  }

  CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, n, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_y, kernel)

  size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
  size_t local = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

  err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0,
  NULL, NULL);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to enqueue kernel '"
               << kernel_name.c_str()
               << "' on GPU "
               << device.name()
               << " : "
               << caffe::OpenCL::what(err);
    return false;
  }

  DLOG(INFO) << "kernel '"
             << kernel_name.c_str()
             << "' executed on GPU "
             << device.name();

  CL_SET_KERNEL_ARG_END

  return true;
}
template bool clsign<float>(
    const int n,
    const void* array_GPU_x,
    void* array_GPU_y);
template bool clsign<double>(
    const int n,
    const void* array_GPU_x,
    void* array_GPU_y);

template<typename T>
bool clsgnbit(const int n, const void* array_GPU_x, void* array_GPU_y) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  std::string kernel_name = clGetKernelName<T>("clsgnbit");

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  cl_kernel* kernel = device.getKernel(kernel_name);
  if (kernel == NULL) {
    return false;
  }

  CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, n, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_y, kernel)

  size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
  size_t local = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

  err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0,
  NULL, NULL);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to enqueue kernel '"
               << kernel_name.c_str()
               << "' on GPU "
               << device.name()
               << " : "
               << caffe::OpenCL::what(err);
    return false;
  }

  DLOG(INFO) << "kernel '"
             << kernel_name.c_str()
             << "' executed on GPU "
             << device.name();

  CL_SET_KERNEL_ARG_END

  return true;
}
template bool clsgnbit<float>(
    const int n,
    const void* array_GPU_x,
    void* array_GPU_y);
template bool clsgnbit<double>(
    const int n,
    const void* array_GPU_x,
    void* array_GPU_y);

template<typename T>
bool clabs(const int n, const void* array_GPU_x, void* array_GPU_y) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  std::string kernel_name = clGetKernelName<T>("clabs");

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  cl_kernel* kernel = device.getKernel(kernel_name);
  if (kernel == NULL) {
    return false;
  }

  CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, n, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_y, kernel)

  size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
  size_t local = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

  err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0,
  NULL, NULL);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to enqueue kernel '"
               << kernel_name.c_str()
               << "' on GPU "
               << device.name()
               << " : "
               << caffe::OpenCL::what(err);
    return false;
  }

  DLOG(INFO) << "kernel '"
             << kernel_name.c_str()
             << "' executed on GPU "
             << device.name();

  CL_SET_KERNEL_ARG_END

  return true;
}
template bool clabs<float>(
    const int n,
    const void* array_GPU_x,
    void* array_GPU_y);
template bool clabs<double>(
    const int n,
    const void* array_GPU_x,
    void* array_GPU_y);

template<typename T>
bool cldiv(
    const int n,
    const void* array_GPU_x,
    const void* array_GPU_y,
    void* array_GPU_z) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  std::string kernel_name = clGetKernelName<T>("cldiv");

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  cl_kernel* kernel = device.getKernel(kernel_name);
  if (kernel == NULL) {
    return false;
  }

  CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, n, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_y, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_z, kernel)

  size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
  size_t local = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

  err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0,
  NULL, NULL);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to enqueue kernel '"
               << kernel_name.c_str()
               << "' on GPU "
               << device.name()
               << " : "
               << caffe::OpenCL::what(err);
    return false;
  }

  DLOG(INFO) << "kernel '"
             << kernel_name.c_str()
             << "' executed on GPU "
             << device.name();

  CL_SET_KERNEL_ARG_END

  return true;
}
template bool cldiv<float>(
    const int n,
    const void* array_GPU_x,
    const void* array_GPU_y,
    void* array_GPU_z);
template bool cldiv<double>(
    const int n,
    const void* array_GPU_x,
    const void* array_GPU_y,
    void* array_GPU_z);

template<typename T>
bool clmul(
    const int n,
    const void* array_GPU_x,
    const void* array_GPU_y,
    void* array_GPU_z) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  std::string kernel_name = clGetKernelName<T>("clmul");

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  cl_kernel* kernel = device.getKernel(kernel_name);
  if (kernel == NULL) {
    return false;
  }

  CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, n, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_y, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_z, kernel)

  size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
  size_t local = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

  err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0,
  NULL, NULL);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to enqueue kernel '"
               << kernel_name.c_str()
               << "' on GPU "
               << device.name()
               << " : "
               << caffe::OpenCL::what(err);
    return false;
  }

  DLOG(INFO) << "kernel '"
             << kernel_name.c_str()
             << "' executed on GPU "
             << device.name();

  CL_SET_KERNEL_ARG_END

  return true;
}
template bool clmul<float>(
    const int n,
    const void* array_GPU_x,
    const void* array_GPU_y,
    void* array_GPU_z);
template bool clmul<double>(
    const int n,
    const void* array_GPU_x,
    const void* array_GPU_y,
    void* array_GPU_z);

template<typename T>
bool clBLASscal(
    const int n,
    const float alpha,
    const void* array_x_virtual,
    void* array_y_virtual) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  const void* array_x_logical;
  if (!clMakeLogical(array_x_virtual, &array_x_logical)) {
    return false;
  }

  const void* array_y_logical;
  if (!clMakeLogical(array_y_virtual, &array_y_logical)) {
    return false;
  }

  float f;
  double d;
  const type_info& floatID   = typeid(f);
  const type_info& doubleID  = typeid(d);

  if (typeid(T) == floatID) {
    if (!CL_CHECK(clblasScopy(n,
                              (cl_mem) array_x_logical, 0, 1,
                              (cl_mem) array_y_logical, 0, 1, 1,
                              queue, 0, NULL, NULL))) {
      LOG(ERROR) << "clblasScopy() failed on GPU " << device.name();
      return false;
    }
    DLOG(INFO) << device.name() << " clblasScopy() succeeded";
    if ( !CL_CHECK(clblasSscal(n,
                               alpha, (cl_mem) array_y_logical, 0, 1, 1,
                               queue, 0, NULL, NULL) ) ) {
      LOG(ERROR) << "clblasSscal() failed on GPU " << device.name();
      return false;
    }
    DLOG(INFO) << device.name() << " clblasSscal() succeeded";
  }

  if (typeid(T) == doubleID) {
    if (!CL_CHECK(clblasDcopy(n,
                              (cl_mem) array_x_logical, 0, 1,
                              (cl_mem) array_y_logical, 0, 1, 1,
                              queue, 0, NULL, NULL))) {
      LOG(ERROR) << "clblasDcopy() failed on GPU " << device.name();
      return false;
    }
    DLOG(INFO) << device.name() << " clblasDcopy() succeeded";
    if ( !CL_CHECK(clblasDscal(n,
                               alpha, (cl_mem) array_y_logical, 0, 1, 1,
                               queue, 0, NULL, NULL) ) ) {
      LOG(ERROR) << "clblasDscal() failed on GPU " << device.name();
      return false;
    }
    DLOG(INFO) << device.name() << " clblasDscal() succeeded";
  }

  return true;
}
template bool clBLASscal<float>(
    const int n,
    const float alpha,
    const void* array_x_virtual,
    void* array_y_virtual);
template bool clBLASscal<double>(
    const int n,
    const float alpha,
    const void* array_x_virtual,
    void* array_y_virtual);

template<typename T>
bool clBLASdot(
    const int n,
    const T* x,
    const int incx,
    const T* y,
    const int incy,
    T* out) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  void* x_base = clGetMemoryBase(x);
  if (!x_base) {
    LOG(ERROR) << "failed to get OpenCL device memory address"
                  " for virtual address @ " << x;
    return false;
  }
  const void* x_device = NULL;
  if (!clMakeLogical(x_base, &x_device)) {
    return false;
  }
  size_t x_offset = clGetMemoryOffset(x);
  if (x_offset < 0) {
    LOG(ERROR) << "failed to get OpenCL device memory offset"
                 " for virtual address @ " << x;
    return false;
  }
  x_offset /= sizeof(T);
  DLOG(INFO) << x
             << " -> "
             << x_base
             << " with offset = "
             << x_offset
             << " on "
             << x_device;

  void* y_base = clGetMemoryBase(y);
  if (!y_base) {
    LOG(ERROR) << "failed to get OpenCL device memory address"
                 " for virtual address @ " << y;
    return false;
  }
  const void* y_device = NULL;
  if (!clMakeLogical(y_base, &y_device)) {
    return false;
  }
  size_t y_offset = clGetMemoryOffset(y);
  if (y_offset < 0) {
    LOG(ERROR) << "failed to get OpenCL device memory offset"
                 " for virtual address @ " << y;
    return false;
  }
  y_offset /= sizeof(T);
  DLOG(INFO) << y
             << " -> "
             << y_base
             << " with offset = "
             << y_offset
             << " on "
             << y_device;

  void* dot;
  if (!caffe::OpenCL::clGetBuffer(&dot, sizeof(T))) {
    LOG(ERROR) << "failed to get buffer.";
  }
  const void* dot_device;
  if (!clMakeLogical(dot, &dot_device)) {
    LOG(ERROR) << "failed to get device";
    caffe::OpenCL::clBufferSetAvailable(dot, sizeof(T));
    return false;
  }

  void* buf;
  if (!caffe::OpenCL::clGetBuffer(&buf, n * sizeof(T))) {
    LOG(ERROR) << "failed to get buffer.";
  }
  const void* buf_device;
  if (!clMakeLogical(buf, &buf_device)) {
    LOG(ERROR) << "failed to get device.";
    caffe::OpenCL::clBufferSetAvailable(buf, n*sizeof(T));
    return false;
  }

  float f;
  double d;
  const type_info& floatID   = typeid(f);
  const type_info& doubleID  = typeid(d);

  if (typeid(T) == floatID) {
    if (!CL_CHECK(clblasSdot(n,
                            (cl_mem) dot_device, 0,
                            (cl_mem) x_device, x_offset, 1,
                            (cl_mem) y_device, y_offset, 1,
                            (cl_mem) buf_device, 1, queue, 0, NULL, NULL))) {
      LOG(ERROR) << "clblasSdot() failed on GPU " << device.name();
      caffe::OpenCL::clBufferSetAvailable(dot, sizeof(T));
      caffe::OpenCL::clBufferSetAvailable(buf, n*sizeof(T));
      return false;
    }
    DLOG(INFO) << "clblasSdot() succeeded on GPU " << device.name();
  }

  if (typeid(T) == doubleID) {
    if (!CL_CHECK(clblasDdot(n,
                            (cl_mem) dot_device, 0,
                            (cl_mem) x_device, x_offset, 1,
                            (cl_mem) y_device, y_offset, 1,
                            (cl_mem) buf_device, 1, queue, 0, NULL, NULL))) {
      LOG(ERROR) << "clblasDdot() failed on GPU " << device.name();
      caffe::OpenCL::clBufferSetAvailable(dot, sizeof(T));
      caffe::OpenCL::clBufferSetAvailable(buf, n*sizeof(T));
      return false;
    }
    DLOG(INFO) << "clblasDdot() succeeded on GPU " << device.name();
  }

  caffe::OpenCL::clMemcpy(out, dot, sizeof(T), caffe::OpenCL::COPY_GPU_TO_CPU);
  caffe::OpenCL::clBufferSetAvailable(dot, sizeof(T));
  caffe::OpenCL::clBufferSetAvailable(buf, n * sizeof(T));

  return true;
}
template bool clBLASdot<float>(
    const int n,
    const float* x,
    const int incx,
    const float* y,
    const int incy,
    float* out);
template bool clBLASdot<double>(
    const int n,
    const double* x,
    const int incx,
    const double* y,
    const int incy,
    double* out);

template<typename T>
bool clBLASgemv(
    const clblasTranspose TransA,
    const int m,
    const int n,
    const T alpha,
    const T* A,
    const T* x,
    const T beta,
    T* y) {
  void* A_base = clGetMemoryBase(A);
  if (!A_base) {
    LOG(ERROR) << "failed to get OpenCL device memory address"
                  " for virtual address @ " << A;
    return false;
  }
  size_t A_offset = clGetMemoryOffset(A);
  if (A_offset < 0) {
    LOG(ERROR) << "failed to get OpenCL device memory offset"
                  " for virtual address @ " << A;
    return false;
  }
  A_offset /= sizeof(T);

  void* x_base = clGetMemoryBase(x);
  if (!x_base) {
    LOG(ERROR) << "failed to get OpenCL device memory address"
                  " for virtual address @ " << x;
    return false;
  }
  size_t x_offset = clGetMemoryOffset(x);
  if (x_offset < 0) {
    LOG(ERROR) << "failed to get OpenCL device memory offset"
                  " for virtual address @ " << x;
    return false;
  }
  x_offset /= sizeof(T);

  void* y_base = clGetMemoryBase(y);
  if (!y_base) {
    LOG(ERROR) << "failed to get OpenCL device memory address"
                  " for virtual address @ " << y;
    return false;
  }
  size_t y_offset = clGetMemoryOffset(y);
  if (y_offset < 0) {
    LOG(ERROR) << "failed to get OpenCL device memory offset"
                  " for virtual address @ " << y;
    return false;
  }
  y_offset /= sizeof(T);

  return clBLASgemv(TransA, m, n, alpha, A, A_offset, x, x_offset, beta, y,
      y_offset);
}
template bool clBLASgemv<float>(
    const clblasTranspose TransA,
    const int m,
    const int n,
    const float alpha,
    const float* A,
    const float* x,
    const float beta,
    float* y);
template bool clBLASgemv<double>(
    const clblasTranspose TransA,
    const int m,
    const int n,
    const double alpha,
    const double* A,
    const double* x,
    const double beta,
    double* y);

template<typename T>
bool clBLASgemv(
    const clblasTranspose TransA,
    const int m,
    const int n,
    const T alpha,
    const T* A,
    const int step_A,
    const T* x,
    const int step_x,
    const T beta,
    T* y,
    const int step_y) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  const void* A_logical;
  if (!clMakeLogical(A, &A_logical)) {
    return false;
  }

  const void* x_logical;
  if (!clMakeLogical(x, &x_logical)) {
    return false;
  }

  const void* y_logical;
  if (!clMakeLogical(y, &y_logical)) {
    return false;
  }

  float f;
  double d;
  const type_info& floatID   = typeid(f);
  const type_info& doubleID  = typeid(d);

  if (typeid(T) == floatID) {
    if (!CL_CHECK(clblasSgemv(clblasRowMajor, TransA, m, n, alpha,
                                (cl_mem) A_logical, step_A, n,
                                (cl_mem) x_logical, step_x, 1, beta,
                                (cl_mem) y_logical, step_y, 1, 1,
                                queue, 0, NULL, NULL))) {
      LOG(ERROR) << "clblasSgemv() failed on GPU " << device.name();
      return false;
    }
    DLOG(INFO) << "clblasSgemv() succeeded on GPU " << device.name();
  }

  if (typeid(T) == doubleID) {
    if (!CL_CHECK(clblasDgemv(clblasRowMajor, TransA, m, n, alpha,
                                (cl_mem) A_logical, step_A, n,
                                (cl_mem) x_logical, step_x, 1, beta,
                                (cl_mem) y_logical, step_y, 1, 1,
                                queue, 0, NULL, NULL))) {
      LOG(ERROR) << "clblasDgemv() failed on GPU " << device.name();
      return false;
    }
    DLOG(INFO) << "clblasDgemv() succeeded on GPU " << device.name();
  }

  return true;
}
template bool clBLASgemv<float>(
    const clblasTranspose TransA,
    const int m,
    const int n,
    const float alpha,
    const float* A,
    const int step_A,
    const float* x,
    const int step_x,
    const float beta,
    float* y,
    const int step_y);
template bool clBLASgemv<double>(
    const clblasTranspose TransA,
    const int m,
    const int n,
    const double alpha,
    const double* A,
    const int step_A,
    const double* x,
    const int step_x,
    const double beta,
    double* y,
    const int step_y);

template<typename T>
bool clgemv(
    const clblasTranspose TransA,
    const int m,
    const int n,
    const T alpha,
    const T* A,
    const T* x,
    const T beta,
    T* y) {
  int M = (TransA == clblasNoTrans) ? m : n;
  int N = (TransA == clblasNoTrans) ? n : m;

  return clgemm(TransA, clblasNoTrans, M, 1, N, alpha, A, x, beta, y,
  NULL);
}
template bool clgemv<float>(
    const clblasTranspose TransA,
    const int m,
    const int n,
    const float alpha,
    const float* A,
    const float* x,
    const float beta,
    float* y);
template bool clgemv<double>(
    const clblasTranspose TransA,
    const int m,
    const int n,
    const double alpha,
    const double* A,
    const double* x,
    const double beta,
    double* y);

template<typename T>
bool clgemv(
    const clblasTranspose TransA,
    const int m,
    const int n,
    const T alpha,
    const T* A,
    const size_t step_A,
    const T* x,
    const size_t step_x,
    const T beta,
    T* y,
    const size_t step_y) {
  int M = (TransA == clblasNoTrans) ? m : n;
  int N = (TransA == clblasNoTrans) ? n : m;

  return clgemm(TransA, clblasNoTrans, M, 1, N, alpha, A, step_A, x, step_x,
      beta, y, step_y,
      NULL);
}
template bool clgemv<float>(
    const clblasTranspose TransA,
    const int m,
    const int n,
    const float alpha,
    const float* A,
    const size_t step_A,
    const float* x,
    const size_t step_x,
    const float beta,
    float* y,
    const size_t step_y);
template bool clgemv<double>(
    const clblasTranspose TransA,
    const int m,
    const int n,
    const double alpha,
    const double* A,
    const size_t step_A,
    const double* x,
    const size_t step_x,
    const double beta,
    double* y,
    const size_t step_y);

template<typename T>
bool clgemm(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const T alpha,
    const T* A,
    const T* B,
    const T beta,
    T* C,
    cl_event* event) {
  void* A_base = clGetMemoryBase(A);
  if (!A_base) {
    LOG(ERROR) << "failed to get OpenCL device memory address"
                  " for virtual address @ " << A;
    return false;
  }
  const void* A_device = NULL;
  if (!clMakeLogical(A_base, &A_device)) {
    return false;
  }
  size_t A_offset = clGetMemoryOffset(A);
  if (A_offset < 0) {
    LOG(ERROR) << "failed to get OpenCL device memory offset"
                  " for virtual address @ " << A;
    return false;
  }
  A_offset /= sizeof(T);
  DLOG(INFO) << A
             << " -> "
             << A_base
             << " with offset = "
             << A_offset
             << " on "
             << A_device;

  void* B_base = clGetMemoryBase(B);
  if (!B_base) {
    LOG(ERROR) << "failed to get OpenCL device memory address"
                  " for virtual address @ " << B;
    return false;
  }
  const void* B_device = NULL;
  if (!clMakeLogical(B_base, &B_device)) {
    return false;
  }
  size_t B_offset = clGetMemoryOffset(B);
  if (B_offset < 0) {
    LOG(ERROR) << "failed to get OpenCL device memory offset"
                  " for virtual address @ " << B;
    return false;
  }
  B_offset /= sizeof(T);
  DLOG(INFO) << B
             << " -> "
             << B_base
             << " with offset = "
             << B_offset
             << " on "
             << B_device;

  void* C_base = clGetMemoryBase(C);
  if (!C_base) {
    LOG(ERROR) << "failed to get OpenCL device memory address"
                  " for virtual address @ " << C;
    return false;
  }
  const void* C_device = NULL;
  if (!clMakeLogical(C_base, &C_device)) {
    return false;
  }
  size_t C_offset = clGetMemoryOffset(C);
  if (C_offset < 0) {
    LOG(ERROR) << "failed to get OpenCL device memory offset"
                  " for virtual address @ " << C;
    return false;
  }
  C_offset /= sizeof(T);
  DLOG(INFO) << C
             << " -> "
             << C_base
             << " with offset = "
             << C_offset
             << " on "
             << C_device;

  return clgemm(TransA, TransB, m, n, k, alpha, reinterpret_cast<T*>(A_base),
      A_offset, reinterpret_cast<T*>(B_base), B_offset, beta,
      reinterpret_cast<T*>(C_base), C_offset, event);
}
template bool clgemm<float>(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const float alpha,
    const float* A,
    const float* B,
    const float beta,
    float* C,
    cl_event* event);
template bool clgemm<double>(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const double alpga,
    const double* A,
    const double* B,
    const double beta,
    double* C,
    cl_event* event);

template<typename T>
int getAlignedSize(const int dataSize, const int blockSize) {
  assert(dataSize > 0);
  assert(blockSize > 0);

  if (dataSize % blockSize == 0) {
    return dataSize;
  }

  return (dataSize / blockSize + 1) * blockSize;
}
template int getAlignedSize<float>(const int dataSize, const int blockSize);
template int getAlignedSize<double>(const int dataSize, const int blockSize);

template<typename T>
bool parameterSearch(
    const size_t* globalSize,
    const size_t* localLimit,
    size_t* localSize) {
  size_t proposedLocalSize = 1;

  while (proposedLocalSize <= localLimit[0]) {
    DLOG(INFO) << "search LOCAL( " << proposedLocalSize << " )";
    if ( globalSize[0] % (proposedLocalSize*2) == 0 ) {
      proposedLocalSize *= 2;
    } else {
      break;
    }
  }
  if (proposedLocalSize > localLimit[0]) {
    proposedLocalSize /= 2;
  }
  localSize[0] = proposedLocalSize;

  DLOG(INFO) << "found LOCAL(" << localSize[0] << " )";
  return true;
}
template bool parameterSearch<float>(
    const size_t* globalSize,
    const size_t* localLimit,
    size_t* localSize);
template bool parameterSearch<double>(
    const size_t* globalSize,
    const size_t* localLimit,
    size_t* localSize);

template<typename T>
bool parameterSearch2D(
    const size_t* globalSize,
    const size_t* localLimit,
    size_t* localSize) {
  size_t proposedLocalSizeX = 1;
  size_t proposedLocalSizeY = 1;
  bool alt = false;
  while (proposedLocalSizeX * proposedLocalSizeY
      <= localLimit[0] * localLimit[1]) {
    DLOG(INFO) << "search LOCAL( "
               << proposedLocalSizeX
               << " , "
               << proposedLocalSizeY
               << " )";

    if ( !alt ) {
      if ( globalSize[0] % (proposedLocalSizeX*2) == 0 ) {
        proposedLocalSizeX *= 2;
      } else {
        break;
      }
    } else {
      if ( globalSize[1] % (proposedLocalSizeY*2) == 0 ) {
        proposedLocalSizeY *= 2;
      } else {
        break;
      }
    }
    alt = !alt;
  }
  if (alt) {
    if (proposedLocalSizeX > localLimit[0]) {
      proposedLocalSizeX /= 2;
    }
  } else {
    if (proposedLocalSizeY > localLimit[1]) {
      proposedLocalSizeY /= 2;
    }
  }

  localSize[0] = proposedLocalSizeX;
  localSize[1] = proposedLocalSizeY;

  DLOG(INFO) << "found LOCAL( "
             << localSize[0]
             << " , "
             << localSize[1]
             << " )";
  return true;
}
template bool parameterSearch2D<float>(
    const size_t* globalSize,
    const size_t* localLimit,
    size_t* localSize);
template bool parameterSearch2D<double>(
    const size_t* globalSize,
    const size_t* localLimit,
    size_t* localSize);

template<typename T>
bool clgemm(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const T alpha,
    const T* A,
    const size_t idx_offset_A,
    const T* B,
    const size_t idx_offset_B,
    const T beta,
    T* C,
    const size_t idx_offset_C,
    cl_event* event) {
  std::tr1::shared_ptr<OpenCLPlatform> pf = OpenCLManager::CurrentPlatform();
  OpenCLDevice& device = pf->CurrentDevice();
  cl_command_queue* queue = device.getCurrentCommandQueue();

  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  std::string kernel_name;
  int numKernelDims = 2;

  size_t* global = reinterpret_cast<size_t*>(malloc(
      numKernelDims * sizeof(size_t)));
  size_t* local = reinterpret_cast<size_t*>(malloc(
      numKernelDims * sizeof(size_t)));
  size_t* limits = reinterpret_cast<size_t*>(malloc(
      numKernelDims * sizeof(size_t)));

  for (int i = 0; i < 1; i++) {
    if (k == 1) {
      // kernel_name = clGetKernelName<T>("mmul_NA_NB_MN1");
      kernel_name = clGetKernelName<T>("mmul_NA_NB_MN1_v2");
      global[0] = getAlignedSize<T>(n, 1);
      local[0] = 1;
      global[1] = getAlignedSize<T>(m, m);
      local[1] = m;
      continue;
    }

    if (OPENCL_BLOCK_SIZE_Y == OPENCL_BLOCK_SIZE_X) {
      /*
       if (TransA == clblasNoTrans && TransB == clblasNoTrans) {
       kernel_name = clGetKernelName<T>("mmul_NA_NB_MNS");
       int workItemSizeM = PS;
       int workItemSizeN = PS;

       int alignedSizeM = getAlignedSize<T>(m, workItemSizeM);
       int alignedSizeN = getAlignedSize<T>(n, workItemSizeN);

       global[0] = alignedSizeN/workItemSizeN;
       global[1] = alignedSizeM/workItemSizeM;

       limits[0] = global[0] > 4 ? 16 : global[0];
       limits[1] = global[1] > 4 ? 16 : global[1];

       parameterSearch2D<T>(global, limits, local);
       int BS = 128;
       global[0] = m;//BS;//getAlignedSize<T>(n, OPENCL_BLOCK_SIZE);
       global[1] = 1;//m/BS;//getAlignedSize<T>(m, OPENCL_BLOCK_SIZE);
       local[0]  = BS;
       local[1]  = 1;
       continue;
       }
       */

      if (TransA == clblasNoTrans && TransB == clblasNoTrans) {
        kernel_name = clGetKernelName<T>("mmul_NA_NB");
        global[0] = getAlignedSize<T>(n, OPENCL_BLOCK_SIZE);
        local[0] = OPENCL_BLOCK_SIZE;
        global[1] = getAlignedSize<T>(m, OPENCL_BLOCK_SIZE);
        local[1] = OPENCL_BLOCK_SIZE;
        continue;
      }
      if (TransA == clblasTrans && TransB == clblasTrans) {
        kernel_name = clGetKernelName<T>("mmul_TA_TB");
        global[0] = getAlignedSize<T>(n, OPENCL_BLOCK_SIZE);
        local[0] = OPENCL_BLOCK_SIZE;
        global[1] = getAlignedSize<T>(m, OPENCL_BLOCK_SIZE);
        local[1] = OPENCL_BLOCK_SIZE;
        continue;
      }
      if (TransA == clblasTrans && TransB == clblasNoTrans) {
        kernel_name = clGetKernelName<T>("mmul_TA_NB");
        global[0] = getAlignedSize<T>(n, OPENCL_BLOCK_SIZE);
        local[0] = OPENCL_BLOCK_SIZE;
        global[1] = getAlignedSize<T>(m, OPENCL_BLOCK_SIZE);
        local[1] = OPENCL_BLOCK_SIZE;
        continue;
      }
      if (TransA == clblasNoTrans && TransB == clblasTrans) {
        kernel_name = clGetKernelName<T>("mmul_NA_TB");
        global[0] = getAlignedSize<T>(n, OPENCL_BLOCK_SIZE);
        local[0] = OPENCL_BLOCK_SIZE;
        global[1] = getAlignedSize<T>(m, OPENCL_BLOCK_SIZE);
        local[1] = OPENCL_BLOCK_SIZE;
        continue;
      }
    }

    if ( OPENCL_BLOCK_SIZE_Y % OPENCL_BLOCK_SIZE_X == 0 ) {
      kernel_name = clGetKernelName<T>("mmul_NA_NB_YmodX");
      global[0] = getAlignedSize<T>(n, OPENCL_BLOCK_SIZE_X);
      local[0] = OPENCL_BLOCK_SIZE_X;
      global[1] = getAlignedSize<T>(m, OPENCL_BLOCK_SIZE_Y);
      local[1] = OPENCL_BLOCK_SIZE_Y;
      continue;
    }
    if ( OPENCL_BLOCK_SIZE_X % OPENCL_BLOCK_SIZE_Y == 0 ) {
      kernel_name = clGetKernelName<T>("mmul_NA_NB_XmodY");
      global[0] = getAlignedSize<T>(n, OPENCL_BLOCK_SIZE_X);
      local[0] = OPENCL_BLOCK_SIZE_X;
      global[1] = getAlignedSize<T>(m, OPENCL_BLOCK_SIZE_Y);
      local[1] = OPENCL_BLOCK_SIZE_Y;
      continue;
    }
  }

  cl_kernel* kernel = device.getKernel(kernel_name);
  if (kernel == NULL) {
    return false;
  }

  CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, m, kernel)
  CL_SET_TYPE_KERNEL_ARG(int, n, kernel)
  CL_SET_TYPE_KERNEL_ARG(int, k, kernel)
  CL_SET_TYPE_KERNEL_ARG(T, alpha, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&A, kernel)
  CL_SET_TYPE_KERNEL_ARG(size_t, idx_offset_A, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&B, kernel)
  CL_SET_TYPE_KERNEL_ARG(size_t, idx_offset_B, kernel)
  CL_SET_TYPE_KERNEL_ARG(T, beta, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&C, kernel)
  CL_SET_TYPE_KERNEL_ARG(size_t, idx_offset_C, kernel)

  DLOG(INFO) << "MNK   = ( " << m << " | " << n << " | " << k << " )";
  DLOG(INFO) << "GSIZE = ( " << global[0] << " | " << global[1] << " )";
  DLOG(INFO) << "LSIZE = ( " << local[0] << " | " << local[1] << " )";

  err = clEnqueueNDRangeKernel(*queue, *kernel, numKernelDims, NULL, global,
      local, 0, NULL, event);
  if (err != CL_SUCCESS) {
    std::ostringstream oss;
    oss << "Failed to enqueue kernel '"
        << kernel_name.c_str()
        << "' on GPU "
        << device.name()
        << " : "
        << what(err);

    LOG(ERROR) << oss.str();
    throw OpenCLSupportException(oss.str());
    return false;
  }
  DLOG(INFO) << "kernel '"
             << kernel_name.c_str()
             << "' executed on GPU "
             << device.name();

  CL_SET_KERNEL_ARG_END
  return true;
}
template bool clgemm<float>(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const float alpha,
    const float* A,
    const size_t idx_offset_A,
    const float* B,
    const size_t idx_offset_B,
    const float beta,
    float* C,
    const size_t idx_offset_C,
    cl_event* event);
template bool clgemm<double>(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const double alpha,
    const double* A,
    const size_t idx_offset_A,
    const double* B,
    const size_t idx_offset_B,
    const double beta,
    double* C,
    const size_t idx_offset_C,
    cl_event* event);

template<typename T>
bool cl_group_gemm(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const int gm,
    const int gn,
    const int gk,
    const T alpha,
    const T* A,
    const size_t idx_offset_A,
    const T* B,
    const size_t idx_offset_B,
    const T beta,
    T* C,
    const size_t idx_offset_C,
    cl_event* event) {
  if (gm != 1) {
    LOG(ERROR) << "partition of M = "
               << m
               << " into GM = "
               << gm
               << " currently not supported";

    return false;
  }
  if ( gk != 1 ) {
    LOG(ERROR) << "partition of K = "
               << m
               << " into GK = "
               << gk
               << " currently not supported";

    return false;
  }
  if ( n % gn != 0 ) {
    LOG(ERROR) << "partition of N = "
               << n
               << " into GN = "
               << gn
               << " must be even such that N % GN = 0 ";

    return false;
  }

  if (TransA != clblasNoTrans && TransB != clblasNoTrans) {
    LOG(WARNING) << "TransA != clblasNoTrans and "
                    "TransB != clblasNoTrans not fully tested";
  }

  std::tr1::shared_ptr<OpenCLPlatform> pf = OpenCLManager::CurrentPlatform();
  OpenCLDevice& device = pf->CurrentDevice();
  cl_command_queue* queue = device.getCurrentCommandQueue();

  if ( !queue ) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  int block_size_x;
  int block_size_y;
  std::string kernel_name;
  int numKernelDims = 0;

  if (TransA == clblasNoTrans && TransB == clblasNoTrans) {
    kernel_name = clGetKernelName<T>("group_mmul_NA_NB");
  }
  if (TransA == clblasTrans && TransB == clblasTrans) {
    kernel_name = clGetKernelName<T>("group_mmul_TA_TB");
  }
  if (TransA == clblasTrans && TransB == clblasNoTrans) {
    kernel_name = clGetKernelName<T>("group_mmul_TA_NB");
  }
  if (TransA == clblasNoTrans && TransB == clblasTrans) {
    kernel_name = clGetKernelName<T>("group_mmul_NA_TB");
  }

  DLOG(INFO) << "kernel = " << kernel_name;

  block_size_x = OPENCL_BLOCK_SIZE;
  block_size_y = OPENCL_BLOCK_SIZE;
  numKernelDims = 2;

  cl_kernel* kernel = device.getKernel(kernel_name);
  if (kernel == NULL) {
    return false;
  }

  CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, m, kernel)
  CL_SET_TYPE_KERNEL_ARG(int, n, kernel)
  CL_SET_TYPE_KERNEL_ARG(int, k, kernel)
  CL_SET_TYPE_KERNEL_ARG(int, gm, kernel)
  CL_SET_TYPE_KERNEL_ARG(int, gn, kernel)
  CL_SET_TYPE_KERNEL_ARG(int, gk, kernel)
  CL_SET_TYPE_KERNEL_ARG(T, alpha, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&A, kernel)
  CL_SET_TYPE_KERNEL_ARG(size_t, idx_offset_A, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&B, kernel)
  CL_SET_TYPE_KERNEL_ARG(size_t, idx_offset_B, kernel)
  CL_SET_TYPE_KERNEL_ARG(T, beta, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&C, kernel)
  CL_SET_TYPE_KERNEL_ARG(size_t, idx_offset_C, kernel)

  size_t* global  = reinterpret_cast<size_t*>(
                      malloc(numKernelDims*sizeof(size_t)));
  size_t* local   = reinterpret_cast<size_t*>(
                      malloc(numKernelDims*sizeof(size_t)));

  int global_x = 0;
  int global_y = 0;

  if ( n % block_size_x == 0 ) {
    global_x = n;
  } else {
    global_x = (n/block_size_x + 1)*block_size_x;
  }

  if ( m % block_size_y == 0 ) {
    global_y = m;
  } else {
    global_y = (m/block_size_y + 1)*block_size_y;
  }

  global[0] = global_x;
  global[1] = global_y;
  local[0] = block_size_x;
  local[1] = block_size_y;

  DLOG(INFO) << "MNK   = ( " << m << " | " << n << " | " << k << " )";
  DLOG(INFO) << "GSIZE = ( " << global[0] << " | " << global[1] << " )";
  DLOG(INFO) << "LSIZE = ( " << local[0] << " | " << local[1] << " )";

  err = clEnqueueNDRangeKernel(*queue, *kernel, numKernelDims, NULL,
                                global, local, 0, NULL, event);
  if ( err != CL_SUCCESS ) {
    std::ostringstream oss;
    oss << "Failed to enqueue kernel '"
        << kernel_name.c_str()
        << "' on GPU "
        << device.name()
        << " : "
        << what(err);

    LOG(ERROR) << oss.str();
    throw OpenCLSupportException(oss.str());
    return false;
  }
  DLOG(INFO) << "kernel '"
             << kernel_name.c_str()
             << "' executed on GPU "
             << device.name();

  CL_SET_KERNEL_ARG_END
  return true;
}
template bool cl_group_gemm<float>(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const int gm,
    const int gn,
    const int gk,
    const float alpha,
    const float* A,
    const size_t idx_offset_A,
    const float* B,
    const size_t idx_offset_B,
    const float beta,
    float* C,
    const size_t idx_offset_C,
    cl_event* event);
template bool cl_group_gemm<double>(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const int gm,
    const int gn,
    const int gk,
    const double alpha,
    const double* A,
    const size_t idx_offset_A,
    const double* B,
    const size_t idx_offset_B,
    const double beta,
    double* C,
    const size_t idx_offset_C,
    cl_event* event);

template<typename T>
bool clBLASgemm(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const T alpha,
    const T* A,
    const T* x,
    const T beta,
    T* y,
    cl_event* event) {
  std::tr1::shared_ptr<OpenCLPlatform> pf = OpenCLManager::CurrentPlatform();
  OpenCLDevice& device = pf->CurrentDevice();

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  void* A_base = clGetMemoryBase(A);
  if (!A_base) {
    LOG(ERROR) << "failed to get OpenCL device memory address"
                  " for virtual address @ " << A;
    return false;
  }
  const void* A_device = NULL;
  if (!clMakeLogical(A_base, &A_device)) {
    return false;
  }
  size_t A_offset = clGetMemoryOffset(A);
  if (A_offset < 0) {
    LOG(ERROR) << "failed to get OpenCL device memory offset"
                  " for virtual address @ " << A;
    return false;
  }
  A_offset /= sizeof(T);
  DLOG(INFO) << A
             << " -> "
             << A_base
             << " with offset = "
             << A_offset
             << " on "
             << A_device;

  void* x_base = clGetMemoryBase(x);
  if (!x_base) {
    LOG(ERROR) << "failed to get OpenCL device memory address"
                  " for virtual address @ " << x;
    return false;
  }

  const void* x_device = NULL;
  if (!clMakeLogical(x_base, &x_device)) {
    return false;
  }
  size_t x_offset = clGetMemoryOffset(x);
  if (x_offset < 0) {
    LOG(ERROR) << "failed to get OpenCL device memory offset"
                  " for virtual address @ " << x;
    return false;
  }
  x_offset /= sizeof(T);
  DLOG(INFO) << x
             << " -> "
             << x_base
             << " with offset = "
             << x_offset
             << " on "
             << x_device;

  void* y_base = clGetMemoryBase(y);
  if (!y_base) {
    LOG(ERROR) << "failed to get OpenCL device memory address"
                  " for virtual address @ " << y;
    return false;
  }
  const void* y_device = NULL;
  if (!clMakeLogical(y_base, &y_device)) {
    return false;
  }
  size_t y_offset = clGetMemoryOffset(y);
  if (y_offset < 0) {
    LOG(ERROR) << "failed to get OpenCL device memory offset"
                  " for virtual address @ " << y;
    return false;
  }
  y_offset /= sizeof(T);
  DLOG(INFO) << y
             << " -> "
             << y_base
             << " with offset = "
             << y_offset
             << " on "
             << y_device;

  // Note that cublas follows fortran order.
  int lda = (TransA == clblasNoTrans) ? k : m;
  int ldb = (TransB == clblasNoTrans) ? n : k;
  int ldc = n;

  float f;
  double d;
  const type_info& floatID   = typeid(f);
  const type_info& doubleID  = typeid(d);

  if (typeid(T) == floatID) {
    if (!CL_CHECK(clblasSgemm(clblasRowMajor, TransA, TransB, m, n, k, alpha,
                              (cl_mem) A_device, A_offset, lda,
                              (cl_mem) x_device, x_offset, ldb,
                              beta,
                              (cl_mem) y_device, y_offset, ldc,
                              1, queue, 0, NULL, event))) {
      LOG(ERROR) << "clblasSgemm() failed on GPU " << device.name();
      return false;
    }
    DLOG(INFO) << "clblasSgemm() succeeded on GPU " << device.name();
  }

  if (typeid(T) == doubleID) {
    if (!CL_CHECK(clblasDgemm(clblasRowMajor, TransA, TransB, m, n, k, alpha,
                              (cl_mem) A_device, A_offset, lda,
                              (cl_mem) x_device, x_offset, ldb,
                              beta,
                              (cl_mem) y_device, y_offset, ldc,
                              1, queue, 0, NULL, event))) {
      LOG(ERROR) << "clblasDgemm() failed on GPU " << device.name();
      return false;
    }
    DLOG(INFO) << "clblasDgemm() succeeded on GPU " << device.name();
  }
  return true;
}
template bool clBLASgemm<float>(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const float alpha,
    const float* A,
    const float* x,
    const float beta,
    float* y,
    cl_event* event);
template bool clBLASgemm<double>(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const double alpha,
    const double* A,
    const double* x,
    const double beta,
    double* y,
    cl_event* event);

template<typename T>
bool clBLASgemm(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const T alpha,
    const T* A,
    const size_t idx_offset_A,
    const T* x,
    const size_t idx_offset_x,
    const T beta,
    T* y,
    const size_t idx_offset_y) {
  return clBLASgemm(TransA, TransB, m, n, k, alpha, A, idx_offset_A, x,
      idx_offset_x, beta, y, idx_offset_y, NULL);
}
template bool clBLASgemm<float>(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const float alpha,
    const float* A,
    const size_t idx_offset_A,
    const float* x,
    const size_t idx_offset_x,
    const float beta,
    float* y,
    const size_t idx_offset_y);
template bool clBLASgemm<double>(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const double alpha,
    const double* A,
    const size_t idx_offset_A,
    const double* x,
    const size_t idx_offset_x,
    const double beta,
    double* y,
    const size_t idx_offset_y);

template<typename T>
bool clBLASgemm(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const T alpha,
    const T* A,
    const size_t idx_offset_A,
    const T* x,
    const size_t idx_offset_x,
    const T beta,
    T* y,
    const size_t idx_offset_y,
    cl_event* event) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  std::vector<cl_mem> sb;
  std::map<const void*, std::pair<void*, size_t> > bm;

  const void* A_logical;
  if (!clMakeLogical2(A, &A_logical, sb, bm)) {
    return false;
  }

  const void* x_logical;
  if (!clMakeLogical2(x, &x_logical, sb, bm)) {
    return false;
  }

  const void* y_logical;
  if (!clMakeLogical2(y, &y_logical, sb, bm)) {
    return false;
  }

  // Note that cublas follows fortran order.
  int lda = (TransA == clblasNoTrans) ? k : m;
  int ldb = (TransB == clblasNoTrans) ? n : k;
  int ldc = n;

  float f;
  double d;
  const type_info& floatID   = typeid(f);
  const type_info& doubleID  = typeid(d);

  // LOG(INFO) << "M x N x K = " << m << " x " << n << " x " << k;
  if (typeid(T) == floatID) {
    if (!CL_CHECK(clblasSgemm(clblasRowMajor, TransA, TransB, m, n, k, alpha,
                              (cl_mem) A_logical, idx_offset_A, lda,
                              (cl_mem) x_logical, idx_offset_x, ldb,
                              beta,
                              (cl_mem) y_logical, idx_offset_y, ldc,
                              1, queue, 0, NULL, event))) {
      LOG(ERROR) << "clblasSgemm() failed on GPU " << device.name();
      return false;
    }
    DLOG(INFO) << "clblasSgemm() succeeded on GPU " << device.name();
  }

  if (typeid(T) == doubleID) {
    if (!CL_CHECK(clblasDgemm(clblasRowMajor, TransA, TransB, m, n, k, alpha,
                              (cl_mem) A_logical, idx_offset_A, lda,
                              (cl_mem) x_logical, idx_offset_x, ldb,
                              beta,
                              (cl_mem) y_logical, idx_offset_y, ldc,
                              1, queue, 0, NULL, event))) {
      LOG(ERROR) << "clblasDgemm() failed on GPU " << device.name();
      return false;
    }
    DLOG(INFO) << "clblasDgemm() succeeded on GPU " << device.name();
  }

  clReleaseBufferMap(bm);
  clReleaseSubBuffers(sb);

  return true;
}
template bool clBLASgemm<float>(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const float alpha,
    const float* A,
    const size_t idx_offset_A,
    const float* x,
    const size_t idx_offset_x,
    const float beta,
    float* y,
    const size_t idx_offset_y,
    cl_event* event);
template bool clBLASgemm<double>(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const double alpha,
    const double* A,
    const size_t idx_offset_A,
    const double* x,
    const size_t idx_offset_x,
    const double beta,
    double* y,
    const size_t idx_offset_y,
    cl_event* event);

template<typename T>
bool clBLASgemm(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const T alpha,
    const T* A,
    const size_t idx_offset_A,
    const size_t lda,
    const T* x,
    const size_t idx_offset_x,
    const size_t ldx,
    const T beta,
    T* y,
    const size_t idx_offset_y,
    const size_t ldy,
    cl_event* event) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  std::vector<cl_mem> sb;
  std::map<const void*, std::pair<void*, size_t> > bm;

  const void* A_logical;
  if (!clMakeLogical2(A, &A_logical, sb, bm)) {
    return false;
  }

  const void* x_logical;
  if (!clMakeLogical2(x, &x_logical, sb, bm)) {
    return false;
  }

  const void* y_logical;
  if (!clMakeLogical2(y, &y_logical, sb, bm)) {
    return false;
  }

  float f;
  double d;
  const type_info& floatID   = typeid(f);
  const type_info& doubleID  = typeid(d);

  // Note that cublas follows fortran order.
  /*
   int lda = (TransA == clblasNoTrans) ? k : m;
   int ldb = (TransB == clblasNoTrans) ? n : k;
   int ldc = n;
   */

  LOG(INFO) << "M x N x K = " << m << " x " << n << " x " << k;
  LOG(INFO) << "LDA x LDB x LDC = " << lda << " x " << ldx << " x " << ldy;
  if (typeid(T) == floatID) {
    if (!CL_CHECK(clblasSgemm(clblasRowMajor, TransA, TransB, m, n, k, alpha,
                              (cl_mem) A_logical, idx_offset_A, lda,
                              (cl_mem) x_logical, idx_offset_x, ldx,
                              beta,
                              (cl_mem) y_logical, idx_offset_y, ldy,
                              1,
                              queue, 0, NULL, event))) {
      LOG(ERROR) << "clblasSgemm() failed on GPU " << device.name();
      return false;
    }
    DLOG(INFO) << "clblasSgemm() succeeded on GPU " << device.name();
  }

  if (typeid(T) == doubleID) {
    if (!CL_CHECK(clblasDgemm(clblasRowMajor, TransA, TransB, m, n, k, alpha,
                              (cl_mem) A_logical, idx_offset_A, lda,
                              (cl_mem) x_logical, idx_offset_x, ldx,
                              beta,
                              (cl_mem) y_logical, idx_offset_y, ldy,
                              1,
                              queue, 0, NULL, event))) {
      LOG(ERROR) << "clblasDgemm() failed on GPU " << device.name();
      return false;
    }
    DLOG(INFO) << "clblasDgemm() succeeded on GPU " << device.name();
  }

  clReleaseBufferMap(bm);
  clReleaseSubBuffers(sb);

  return true;
}
template bool clBLASgemm<float>(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const float alpha,
    const float* A,
    const size_t idx_offset_A,
    const size_t lda,
    const float* x,
    const size_t idx_offset_x,
    const size_t ldx,
    const float beta,
    float* y,
    const size_t idx_offset_y,
    const size_t ldy,
    cl_event* event);
template bool clBLASgemm<double>(
    const clblasTranspose TransA,
    const clblasTranspose TransB,
    const int m,
    const int n,
    const int k,
    const double alpha,
    const double* A,
    const size_t idx_offset_A,
    const size_t lda,
    const double* x,
    const size_t idx_offset_x,
    const size_t ldx,
    const double beta,
    double* y,
    const size_t idx_offset_y,
    const size_t ldy,
    cl_event* event);

template<typename T>
bool clBLASaxpy(
    const int N,
    const T alpha,
    const T* X,
    const int incr_x,
    T* Y,
    const int incr_y) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  void* X_base = clGetMemoryBase(X);
  if (!X_base) {
    LOG(ERROR) << "failed to get OpenCL device memory address"
                  " for virtual address @ " << X;
    return false;
  }
  const void* X_device = NULL;
  if (!clMakeLogical(X_base, &X_device)) {
    return false;
  }
  size_t X_offset = clGetMemoryOffset(X);
  if (X_offset < 0) {
    LOG(ERROR) << "failed to get OpenCL device memory offset"
                  " for virtual address @ " << X;
    return false;
  }
  X_offset /= sizeof(T);
  DLOG(INFO) << X
             << " -> "
             << X_base
             << " with offset = "
             << X_offset
             << " on "
             << X_device;

  void* Y_base = clGetMemoryBase(Y);
  if (!Y_base) {
    LOG(ERROR) << "failed to get OpenCL device memory address"
                  " for virtual address @ " << Y;
    return false;
  }
  const void* Y_device = NULL;
  if (!clMakeLogical(Y_base, &Y_device)) {
    return false;
  }
  size_t Y_offset = clGetMemoryOffset(Y);
  if (Y_offset < 0) {
    LOG(ERROR) << "failed to get OpenCL device memory offset"
                  " for virtual address @ " << Y;
    return false;
  }
  Y_offset /= sizeof(T);
  DLOG(INFO) << Y
             << " -> "
             << Y_base
             << " with offset = "
             << Y_offset
             << " on "
             << Y_device;

  float f;
  double d;
  const type_info& floatID   = typeid(f);
  const type_info& doubleID  = typeid(d);

  if (typeid(T) == floatID) {
    if (!CL_CHECK(clblasSaxpy(N, alpha,
                              (cl_mem) X_device, X_offset, incr_x,
                              (cl_mem) Y_device, Y_offset, incr_y,
                              1,
                              queue, 0, NULL, NULL))) {
      LOG(ERROR) << "clblasSaxpy() failed on GPU " << device.name();
      return false;
    }
    DLOG(INFO) << "clblasSaxpy() succeeded on GPU " << device.name();
  }

  if (typeid(T) == doubleID) {
    if (!CL_CHECK(clblasDaxpy(N, alpha,
                              (cl_mem) X_device, X_offset, incr_x,
                              (cl_mem) Y_device, Y_offset, incr_y,
                              1,
                              queue, 0, NULL, NULL))) {
      LOG(ERROR) << "clblasDaxpy() failed on GPU " << device.name();
      return false;
    }
    DLOG(INFO) << "clblasDaxpy() succeeded on GPU " << device.name();
  }

  return true;
}
template bool clBLASaxpy<float>(
    const int N,
    const float alpha,
    const float* X,
    const int incr_x,
    float* Y,
    const int incr_y);
template bool clBLASaxpy<double>(
    const int N,
    const double alpha,
    const double* X,
    const int incr_x,
    double* Y,
    const int incr_y);

bool clIsVirtualMemory(const void* p) {
  return caffe::OpenCLMemory::isHighMem(p);
}

bool clMakeLogical(const void* ptr_virtual, const void** ptr_logical) {
  if (mapMemoryToDevice.find(ptr_virtual) != mapMemoryToDevice.end()) {
    *ptr_logical = mapMemoryToDevice[ptr_virtual];
    return true;
  }

  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  if (!clIsVirtualMemory(ptr_virtual)) {
    LOG(WARNING)<< "PTR@" << ptr_virtual << " is not in virtual memory.";
    *ptr_logical = ptr_virtual;
    mapMemoryToDevice[ptr_virtual] = ptr_virtual;
    return true;
  }

  OpenCLMemory* clMem;
  if (!device.get(ptr_virtual, &clMem)) {
    LOG(ERROR) << "failed to get OpenCLMemory object associated with VM@"
               << ptr_virtual;
    return false;
  }
  *ptr_logical = clMem->getLogicalPointer();

  if (clIsVirtualMemory(*ptr_logical)) {
    DLOG(INFO) << "failed to convert VM@"
               << ptr_virtual
               << " from virtual to logical address space.";
    return false;
  }
  mapMemoryToDevice[ptr_virtual] = *ptr_logical;

  return true;
}

bool clMakeLogical2(
    const void* ptr_virtual,
    const void** ptr_logical,
    std::vector<cl_mem>& subBuffers,  // NOLINT(*)
    std::map<const void*, std::pair<void*, size_t> >& bufferMap) {  // NOLINT(*)
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  if (!clIsVirtualMemory(ptr_virtual)) {
    LOG(WARNING)<< "PTR@" << ptr_virtual << " is not in virtual memory.";
    *ptr_logical = ptr_virtual;
    return true;
  }

  OpenCLMemory* clMem;
  if (!device.get(ptr_virtual, &clMem)) {
    LOG(ERROR) << "failed to get OpenCLMemory object associated with VM@"
               << ptr_virtual;
    return false;
  }
  *ptr_logical = clMem->getLogicalPointer();

  if (clIsVirtualMemory(*ptr_logical)) {
    LOG(ERROR) << "failed to convert VM@"
               << ptr_virtual
               << " from virtual to logical address space.";
    return false;
  }

  size_t offset = clGetMemoryOffset(ptr_virtual);
  size_t size = clGetMemorySize(ptr_virtual);

  // no offset, no additional steps needed
  if (offset == 0) {
    return true;
  }

  cl_int err;
  cl_uint bytes = device.getDeviceMemBaseAddrAlign();

  // the offset is aligned and can be treated with clCreateSubBuffer
  if (offset % bytes == 0) {
    LOG(WARNING) << "aligned memory offset = "
                 << offset
                 << " detected, create sub-buffer ["
                 << offset
                 << "|"
                 << size-offset
                 << "]";

    cl_buffer_region region = {offset, size - offset};
    *ptr_logical = (const void*) clCreateSubBuffer(
                                    (cl_mem) *ptr_logical,
                                    CL_MEM_READ_WRITE,
                                    CL_BUFFER_CREATE_TYPE_REGION,
                                    &region, &err);
    if ( err != CL_SUCCESS ) {
      LOG(ERROR) << "failed to create sub-buffer on GPU "
                 << device.name()
                 << " : "
                 << what(err);
      return false;
    }
    DLOG(INFO) << "create sub-buffer ";
    subBuffers.push_back((cl_mem) *ptr_logical);
    return true;
  }

  // the offset is not aligned and an additional buffer is needed
  LOG(WARNING) << "sub-buffer memory offset ("
               << offset
               << " Byte) is not aligned with device memory offset ("
               << bytes
               << " Byte)";

  cl_mem_flags flags;
  err = clGetMemObjectInfo((cl_mem) *ptr_logical, CL_MEM_FLAGS, sizeof(flags),
      &flags, NULL);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "failed to query memory properties.";
    return false;
  }

  void* ptr_buffer;
  size_t buffer_size = size - offset;
  if (!clMalloc(&ptr_buffer, buffer_size)) {
    return false;
  }

  if (!clMemcpy(ptr_buffer, ptr_virtual, buffer_size,
      caffe::OpenCL::COPY_DEFAULT)) {
    return false;
  }

  if (!clMakeLogical(ptr_buffer, ptr_logical)) {
    return false;
  }
  bufferMap[ptr_virtual] = std::make_pair(ptr_buffer, buffer_size);
  return true;
}

size_t clGetMemoryOffset(const void* ptr_virtual) {
  if (mapMemoryToOffset.find(ptr_virtual) != mapMemoryToOffset.end()) {
    return mapMemoryToOffset[ptr_virtual];
  }

  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  if (!clIsVirtualMemory(ptr_virtual)) {
    LOG(WARNING)<< "PTR@" << ptr_virtual << " is not in virtual memory.";
    return -1;
  }

  OpenCLMemory* clMem;
  if (!device.get(ptr_virtual, &clMem)) {
    LOG(ERROR) << "failed to get OpenCLMemory object associated with VM@"
               << ptr_virtual;
    return -1;
  }
  size_t mem_offset = static_cast<const char*>(ptr_virtual)
      - static_cast<const char*>(clMem->getVirtualPointer());
  mapMemoryToOffset[ptr_virtual] = mem_offset;

  return mem_offset;
}

size_t clGetMemorySize(const void* ptr_virtual) {
  if (mapMemoryToSize.find(ptr_virtual) != mapMemoryToSize.end()) {
    return mapMemoryToSize[ptr_virtual];
  }

  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  if (!clIsVirtualMemory(ptr_virtual)) {
    LOG(WARNING)<< "PTR@" << ptr_virtual << " is not in virtual memory.";
    return -1;
  }

  OpenCLMemory* clMem;
  if (!device.get(ptr_virtual, &clMem)) {
    LOG(ERROR) << "failed to get OpenCLMemory object associated with VM@"
               << ptr_virtual;
    return -1;
  }
  mapMemoryToSize[ptr_virtual] = clMem->getSize();

  return clMem->getSize();
}

void* clGetMemoryBase(const void* ptr_virtual) {
  if (mapMemoryToBase.find(ptr_virtual) != mapMemoryToBase.end()) {
    return mapMemoryToBase[ptr_virtual];
  }

  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  if (!clIsVirtualMemory(ptr_virtual)) {
    LOG(WARNING)<< "PTR@" << ptr_virtual << " is not in virtual memory.";
    return NULL;
  }

  OpenCLMemory* clMem;
  if (!device.get(ptr_virtual, &clMem)) {
    LOG(ERROR) << "failed to get OpenCLMemory object associated with VM@"
               << ptr_virtual;
    return NULL;
  }

  mapMemoryToBase[ptr_virtual] = clMem->getVirtualPointer();

  return clMem->getVirtualPointer();
}

bool clGetMemoryObject(const void* ptr_virtual, OpenCLMemory** clMem) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  if (!clIsVirtualMemory(ptr_virtual)) {
    LOG(WARNING)<< "PTR@" << ptr_virtual << " is not in virtual memory.";
    return false;
  }

  if (!device.get(ptr_virtual, clMem)) {
    LOG(ERROR) << "failed to get OpenCLMemory object associated with VM@"
               << ptr_virtual;
    return false;
  }
  return true;
}

template<typename T>
bool clsub(
    const int n,
    const T* array_GPU_x,
    const T* array_GPU_y,
    T* array_GPU_z) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  std::string kernel_name = clGetKernelName<T>("clsub");

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  cl_kernel* kernel = device.getKernel(kernel_name);
  if (kernel == NULL) {
    return false;
  }

  CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, n, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_y, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_z, kernel)

  size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
  size_t local = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

  err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0,
  NULL, NULL);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to enqueue kernel '"
               << kernel_name.c_str()
               << "' on GPU "
               << device.name()
               << " : "
               << caffe::OpenCL::what(err);
    return false;
  }

  DLOG(INFO) << "kernel '"
             << kernel_name.c_str()
             << "' executed on GPU "
             << device.name();

  CL_SET_KERNEL_ARG_END

  return true;
}
template bool clsub<float>(
    const int n,
    const float* array_GPU_x,
    const float* array_GPU_y,
    float* array_GPU_z);
template bool clsub<double>(
    const int n,
    const double* array_GPU_x,
    const double* array_GPU_y,
    double* array_GPU_z);

template<typename T>
bool cladd(
    const int n,
    const T* array_GPU_x,
    const T* array_GPU_y,
    T* array_GPU_z) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  std::string kernel_name = clGetKernelName<T>("cladd");

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  cl_kernel* kernel = device.getKernel(kernel_name);
  if (kernel == NULL) {
    return false;
  }

  CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, n, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_y, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_z, kernel)

  size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
  size_t local = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

  err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0,
  NULL, NULL);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to enqueue kernel '"
               << kernel_name.c_str()
               << "' on GPU "
               << device.name()
               << " : "
               << caffe::OpenCL::what(err);
    return false;
  }

  DLOG(INFO) << "kernel '"
             << kernel_name.c_str()
             << "' executed on GPU "
             << device.name();

  CL_SET_KERNEL_ARG_END

  return true;
}
template bool cladd<float>(
    const int n,
    const float* array_GPU_x,
    const float* array_GPU_y,
    float* array_GPU_z);
template bool cladd<double>(
    const int n,
    const double* array_GPU_x,
    const double* array_GPU_y,
    double* array_GPU_z);

template<typename T>
bool cladd_scalar(const int N, const T alpha, T* Y) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  std::string kernel_name = clGetKernelName<T>("cladd_scalar");

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  cl_kernel* kernel = device.getKernel(kernel_name);
  if (kernel == NULL) {
    return false;
  }

  CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, N, kernel)
  CL_SET_TYPE_KERNEL_ARG(T, alpha, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&Y, kernel)

  size_t global = CAFFE_GET_GLOBAL_WORKITEMS(N, OPENCL_LOCAL_SIZE);
  size_t local = CAFFE_GET_LOCAL_WORKITEMS(N, OPENCL_LOCAL_SIZE);

  err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0,
  NULL, NULL);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to enqueue kernel '"
               << kernel_name.c_str()
               << "' on GPU "
               << device.name()
               << " : "
               << caffe::OpenCL::what(err);
    return false;
  }

  DLOG(INFO) << "kernel '"
             << kernel_name.c_str()
             << "' executed on GPU "
             << device.name();

  CL_SET_KERNEL_ARG_END

  return true;
}
template bool cladd_scalar<float>(const int N, const float alpha, float* Y);
template bool cladd_scalar<double>(const int N, const double alpha, double* Y);

template<typename T>
bool clpowx(const int n, const T* array_GPU_x, const T alpha, T* array_GPU_z) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  std::string kernel_name = clGetKernelName<T>("clpowx");

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  cl_kernel* kernel = device.getKernel(kernel_name);
  if (kernel == NULL) {
    return false;
  }

  CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, n, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x, kernel)
  CL_SET_TYPE_KERNEL_ARG(T, alpha, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_z, kernel)

  size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
  size_t local = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

  err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0,
  NULL, NULL);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to enqueue kernel '"
               << kernel_name.c_str()
               << "' on GPU "
               << device.name()
               << " : "
               << caffe::OpenCL::what(err);
    return false;
  }

  DLOG(INFO) << "kernel '"
             << kernel_name.c_str()
             << "' executed on GPU "
             << device.name();

  CL_SET_KERNEL_ARG_END

  return true;
}
template bool clpowx<float>(
    const int n,
    const float* array_GPU_x,
    const float alpha,
    float* array_GPU_z);
template bool clpowx<double>(
    const int n,
    const double* array_GPU_x,
    const double alpha,
    double* array_GPU_z);

template<typename T>
bool clexp(const int n, const T* array_GPU_x, T* array_GPU_y) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

  std::string kernel_name = clGetKernelName<T>("clexp");

  cl_command_queue* queue = device.getCurrentCommandQueue();
  if (!queue) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
    return false;
  }

  cl_kernel* kernel = device.getKernel(kernel_name);
  if (kernel == NULL) {
    return false;
  }

  CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, n, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&array_GPU_y, kernel)

  size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
  size_t local = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

  err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0,
  NULL, NULL);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to enqueue kernel '"
               << kernel_name.c_str()
               << "' on GPU "
               << device.name()
               << " : "
               << caffe::OpenCL::what(err);
    return false;
  }

  DLOG(INFO) << "kernel '"
             << kernel_name.c_str()
             << "' executed on GPU "
             << device.name();

  CL_SET_KERNEL_ARG_END

  return true;
}
template bool clexp<float>(
    const int n,
    const float* array_GPU_x,
    float* array_GPU_z);
template bool clexp<double>(
    const int n,
    const double* array_GPU_x,
    double* array_GPU_z);

bool cl_caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  size_t bytes;

  bytes = n * sizeof(double);
  double* fBuffer = reinterpret_cast<double*>(malloc(bytes));
  if (fBuffer == NULL) {
    LOG(ERROR) << "failed to allocate cpu memory for buffer of "
               << bytes << " Bytes.";
    return false;
  }

  bytes = n * sizeof(unsigned int);
  unsigned int* buffer = (unsigned int*) malloc(bytes);
  if (buffer == NULL) {
    LOG(ERROR) << "failed to allocate cpu memory for buffer of "
               << bytes
               << " Bytes.";
    return false;
  }

  caffe_rng_uniform(n, 0.0, static_cast<double>(UINT_MAX), fBuffer);
  for (int i = 0; i < n; i++) {
    buffer[i] = (unsigned int) fBuffer[i];
  }

  BOOL_CHECK(caffe::OpenCL::clMemcpy(r, buffer, bytes, COPY_CPU_TO_GPU));
  free(buffer);
  free(fBuffer);
  LOG(WARNING) << "caffe_gpu_rng_gaussian() was executed on the CPU"
                  " and random array copied back to GPU.";

  return true;
}

template<typename T>
bool cl_caffe_gpu_rng_uniform(const int n, const T a, const T b, T* r) {
  size_t bytes = n * sizeof(T);
  T* buffer = reinterpret_cast<T*>(malloc(bytes));
  if (buffer == NULL) {
    LOG(ERROR) << "failed to allocate cpu memory for buffer of "
               << bytes
               << " Bytes.";
    return false;
  }

  caffe_rng_uniform(n, a, b, buffer);
  BOOL_CHECK(caffe::OpenCL::clMemcpy(r, buffer, bytes, COPY_CPU_TO_GPU));
  free(buffer);
  LOG(WARNING) << "caffe_gpu_rng_uniform() was executed on the CPU"
                  " and random array copied back to GPU.";

  return true;
}
template bool cl_caffe_gpu_rng_uniform<float>(
    const int n,
    const float a,
    const float b,
    float* r);
template bool cl_caffe_gpu_rng_uniform<double>(
    const int n,
    const double a,
    const double b,
    double* r);

template<typename T>
bool cl_caffe_gpu_rng_gaussian(const int n, const T mu, const T sigma, T* r) {
  size_t bytes = n * sizeof(T);
  T* buffer = reinterpret_cast<T*>(malloc(bytes));
  if (buffer == NULL) {
    LOG(ERROR) << "failed to allocate cpu memory for buffer of "
               << bytes
               << " Bytes.";
    return false;
  }

  caffe_rng_gaussian(n, mu, sigma, buffer);
  BOOL_CHECK(caffe::OpenCL::clMemcpy(r, buffer, bytes, COPY_CPU_TO_GPU));
  free(buffer);
  LOG(WARNING) << "caffe_gpu_rng_gaussian() was executed on the CPU"
                  " and random array copied back to GPU.";

  return true;
}
template bool cl_caffe_gpu_rng_gaussian<float>(
    const int n,
    const float mu,
    const float sigma,
    float* r);
template bool cl_caffe_gpu_rng_gaussian<double>(
    const int n,
    const double mu,
    const double sigma,
    double* r);

template<typename T1, typename T2>
bool cl_caffe_gpu_rng_bernoulli(const int n, const T1 p, T2* r) {
  size_t bytes = n * sizeof(T2);
  T2* buffer = reinterpret_cast<T2*>(malloc(bytes));
  if (buffer == NULL) {
    LOG(ERROR) << "failed to allocate cpu memory for buffer of "
               << bytes
               << " Bytes.";
    return false;
  }

  caffe_rng_bernoulli(n, p, buffer);
  BOOL_CHECK(caffe::OpenCL::clMemcpy(r, buffer, bytes, COPY_CPU_TO_GPU));
  free(buffer);
  LOG(WARNING)<< "caffe_gpu_rng_bernoulli() was executed on the CPU"
                 " and random array copied back to GPU.";

  return true;
}
template bool cl_caffe_gpu_rng_bernoulli<float, int>(
    const int n,
    const float p,
    int* r);
template bool cl_caffe_gpu_rng_bernoulli<double, int>(
    const int n,
    const double p,
    int* r);
template bool cl_caffe_gpu_rng_bernoulli<float, unsigned int>(
    const int n,
    const float p,
    unsigned int* r);
template bool cl_caffe_gpu_rng_bernoulli<double, unsigned int>(
    const int n,
    const double p,
    unsigned int* r);

void clProfile(cl_event event, clProfileResult* profile) {
#ifndef OPENCL_PROFILING
  LOG(WARNING)<< "OpenCL profiling not enabled, "
                 "use -DENABLE_PROFILING at the command line with cmake.";
#else
  caffe::Caffe::DeviceSync();

  cl_ulong t_queued;
  cl_ulong t_submit;
  cl_ulong t_start;
  cl_ulong t_end;

  while (true) {
    cl_int err;

    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,
                                  sizeof(cl_ulong), &t_queued, NULL);
    if ( err != CL_SUCCESS ) {
      LOG(ERROR) << "failed to get profiling information from event: "
                 << caffe::OpenCL::what(err);
      break;
    }
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT,
                                  sizeof(cl_ulong), &t_submit, NULL);
    if ( err != CL_SUCCESS ) {
      LOG(ERROR) << "failed to get profiling information from event: "
                 << caffe::OpenCL::what(err);
      break;
    }
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                  sizeof(cl_ulong), &t_start, NULL);
    if ( err != CL_SUCCESS ) {
      LOG(ERROR) << "failed to get profiling information from event: "
                 << caffe::OpenCL::what(err);
      break;
    }
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                  sizeof(cl_ulong), &t_end, NULL);
    if ( err != CL_SUCCESS ) {
      LOG(ERROR) << "failed to get profiling information from event: "
                 << caffe::OpenCL::what(err);
      break;
    }

    profile->queued.t_bgn = t_queued;
    profile->queued.t_end = t_submit;
    profile->queued.time.ns = profile->queued.t_end - profile->queued.t_bgn;
    profile->queued.time.us = profile->queued.time.ns/1000.0;
    profile->queued.time.ms = profile->queued.time.us/1000.0;
    profile->queued.time.s = profile->queued.time.ms/1000.0;

    profile->submit.t_bgn = t_submit;
    profile->submit.t_end = t_start;
    profile->submit.time.ns = profile->submit.t_end - profile->submit.t_bgn;
    profile->submit.time.us = profile->submit.time.ns/1000.0;
    profile->submit.time.ms = profile->submit.time.us/1000.0;
    profile->submit.time.s = profile->submit.time.ms/1000.0;

    profile->kernel.t_bgn = t_start;
    profile->kernel.t_end = t_end;
    profile->kernel.time.ns = profile->kernel.t_end - profile->kernel.t_bgn;
    profile->kernel.time.us = profile->kernel.time.ns/1000.0;
    profile->kernel.time.ms = profile->kernel.time.us/1000.0;
    profile->kernel.time.s = profile->kernel.time.ms/1000.0;

    profile->call.t_bgn = t_queued;
    profile->call.t_end = t_end;
    profile->call.time.ns = profile->call.t_end - profile->call.t_bgn;
    profile->call.time.us = profile->call.time.ns/1000.0;
    profile->call.time.ms = profile->call.time.us/1000.0;
    profile->call.time.s = profile->call.time.ms/1000.0;

    if ( profile->kernel.compute.Flop != 0.0 ) {
      profile->kernel.compute.Flops =
          profile->kernel.compute.Flop / profile->kernel.time.s;
      profile->kernel.compute.MFlops =
          profile->kernel.compute.Flops / 1000000;
      profile->kernel.compute.GFlops =
          profile->kernel.compute.Flops / 1000000000;
      profile->kernel.compute.TFlops =
          profile->kernel.compute.Flops / 1000000000000;
    }

    if ( profile->kernel.io.Byte != 0.0 ) {
      profile->kernel.io.Bytes =
          profile->kernel.io.Byte / profile->kernel.time.s;
      profile->kernel.io.MBytes =
          profile->kernel.io.Bytes / 1000000;
      profile->kernel.io.GBytes =
          profile->kernel.io.Bytes / 1000000000;
      profile->kernel.io.TBytes =
          profile->kernel.io.Bytes / 1000000000000;
    }
    DLOG(INFO) << "PROFILE(caffe_gpu_gemm()) "
               << "queued( " << F2(profile->queued.time.us) << "us ) "
               << "submit( " << F2(profile->submit.time.us) << "us ) "
               << "kernel( " << F2(profile->kernel.time.us) << "us ) "
               << "all( " << F2(profile->call.time.us) << "us ) "
               << "compute( "
               << F2(profile->kernel.compute.GFlops) << " GFlop/s) "
               << "io( " << F2(profile->kernel.io.GBytes) << " GByte/s)";
    break;
  }
#endif
}

void clLogProfile(
    clProfileResult profile,
    std::string file,
    std::string src,
    std::string function,
    std::string type) {
  // check on file
  bool file_is_new = true;
  if (std::ifstream(file.c_str())) {
    file_is_new = false;
  }

  // open file output stream
  std::ofstream ofs;
  ofs.open(file.c_str(), std::ios::app);
  if (!ofs.is_open()) {
    LOG(ERROR) << "failed to open file '" << file.c_str() << "' for writing.";
    return;
  }

  if (file_is_new) {
    ofs << "\"Device\", " << "\"SDK\", " << "\"NCQ\", " << "\"File\", "
        << "\"Function\", " << "\"DataType\", " << "\"Queued[us]\", "
        << "\"Submit[us]\", " << "\"Kernel[us]\", " << "\"Call[us]\", "
        << "\"FLOP[#]\", " << "\"BYTE[#]\"" << std::endl;
  }

  std::string device;
  std::string sdk;

  if (Caffe::mode() == Caffe::CPU) {
    device = "CPU";
    if (OpenCLManager::CurrentPlatform()->getNumDevices(CL_DEVICE_TYPE_CPU)
        > 0) {
      if (OpenCLManager::CurrentPlatform()->getDevice(CL_DEVICE_TYPE_CPU,
          0) != NULL) {
        device = OpenCLManager::CurrentPlatform()->getDevice(
        CL_DEVICE_TYPE_CPU, 0)->name();
      }
    }
    sdk = "C++";
  }
  if (Caffe::mode() == Caffe::GPU) {
    device =
     OpenCLManager::CurrentPlatform()->getDevice(CL_DEVICE_TYPE_GPU, 0)->name();
    sdk = OpenCLManager::CurrentPlatform()->version();
  }

  ofs << device.c_str() << ", ";
  ofs << sdk.c_str() << ", ";
  ofs << OPENCL_NUM_COMMAND_QUEUES << ", ";
  ofs << src.c_str() << ", ";
  ofs << function.c_str() << ", ";
  ofs << type.c_str() << ", ";
  ofs << profile.queued.time.us << ", ";
  ofs << profile.submit.time.us << ", ";
  ofs << profile.kernel.time.us << ", ";
  ofs << profile.call.time.us << ", ";
  ofs << profile.kernel.compute.Flop << ", ";
  ofs << profile.kernel.io.Byte;
  ofs << std::endl;

  ofs.close();
}

}  // namespace OpenCL

}  // namespace caffe

#endif  // USE_OPENCL
