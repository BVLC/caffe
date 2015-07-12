#ifdef USE_OPENCL

#include <CL/cl.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <glog/logging.h>
#include <caffe/util/OpenCL/OpenCLDevice.hpp>
#include <caffe/util/OpenCL/OpenCLSupport.hpp>
#include <caffe/util/OpenCL/OpenCLParser.hpp>
#include <caffe/util/OpenCL/OpenCLMemory.hpp>
#include <tr1/memory>

#include <fstream>

namespace caffe {

OpenCLDevice::OpenCLDevice() {

	deviceID 					= NULL;
	platformID 					= NULL;
	deviceType 					= 0;
	deviceMaxComputeUnits 		= 0;
	deviceMaxWorkItemDims 		= 0;
	deviceMaxWorkGroupSize 		= 0;
	deviceMaxClockFreqMHz 		= 0;
	deviceMaxMemAllocSize 		= 0;
	deviceMaxArgListSize 		= 0;
	deviceMemAddrAlign 			= 0;
	deviceMinMemAddrAlign 		= 0;
	deviceGlobalMemCacheType 	= 0;
	deviceGlobalMemCacheSize 	= 0;
	deviceGlobalMemCacheLineSize = 0;
	deviceGlobalMemSize 		= 0;
	deviceLocalMemType 			= 0;
	deviceLocalMemSize 			= 0;
	deviceHostUnifiedMem 		= 0;
	deviceMemBaseAddrAlign		= 0;

	for( int i = 0; i < OPENCL_NUM_INPUT_QUEUES; i++ ) {
	  inputQueues[i] = NULL;
	}
  currentInputQueueIdx  = 0;

  for( int i = 0; i < OPENCL_NUM_COMMAND_QUEUES; i++ ) {
    commandQueues[i] = NULL;
  }
  currentCommandQueueIdx  = 0;

  for( int i = 0; i < OPENCL_NUM_OUTPUT_QUEUES; i++ ) {
    outputQueues[i] = NULL;
  }
  currentOutputQueueIdx   = 0;
}

OpenCLDevice::OpenCLDevice(cl_platform_id pid, cl_device_id did) {
	deviceID = did;
	platformID = pid;
	deviceType = 0;
	deviceMaxComputeUnits = 0;
	deviceMaxWorkItemDims = 0;
	deviceMaxWorkGroupSize = 0;
	deviceMaxClockFreqMHz = 0;
	deviceMaxMemAllocSize = 0;
	deviceMaxArgListSize = 0;
	deviceMemAddrAlign = 0;
	deviceMinMemAddrAlign = 0;
	deviceGlobalMemCacheType = 0;
	deviceGlobalMemCacheSize = 0;
	deviceGlobalMemCacheLineSize = 0;
	deviceGlobalMemSize = 0;
	deviceLocalMemType = 0;
	deviceLocalMemSize = 0;
	deviceHostUnifiedMem = 0;
	deviceMemBaseAddrAlign		= 0;

  for( int i = 0; i < OPENCL_NUM_INPUT_QUEUES; i++ ) {
    inputQueues[i] = NULL;
  }
  currentInputQueueIdx  = 0;

  for( int i = 0; i < OPENCL_NUM_COMMAND_QUEUES; i++ ) {
    commandQueues[i] = NULL;
  }
  currentCommandQueueIdx  = 0;

  for( int i = 0; i < OPENCL_NUM_OUTPUT_QUEUES; i++ ) {
    outputQueues[i] = NULL;
  }
  currentOutputQueueIdx   = 0;
}

OpenCLDevice::OpenCLDevice(const OpenCLDevice& dev) {
	deviceID = dev.deviceID;
	platformID = dev.platformID;
	deviceType = dev.deviceType;
	deviceTypeStr = dev.deviceTypeStr;
	deviceMaxComputeUnits = dev.deviceMaxComputeUnits;
	deviceMaxWorkItemDims = dev.deviceMaxWorkItemDims;
	deviceMaxWorkGroupSize = dev.deviceMaxWorkGroupSize;
	deviceMaxClockFreqMHz = dev.deviceMaxClockFreqMHz;
	deviceMaxMemAllocSize = dev.deviceMaxMemAllocSize;
	deviceMaxArgListSize = dev.deviceMaxArgListSize;
	deviceMemAddrAlign = dev.deviceMemAddrAlign;
	deviceMinMemAddrAlign = dev.deviceMinMemAddrAlign;
	deviceGlobalMemCacheType = dev.deviceGlobalMemCacheType;
	deviceGlobalMemCacheSize = dev.deviceGlobalMemCacheSize;
	deviceGlobalMemCacheLineSize = dev.deviceGlobalMemCacheLineSize;
	deviceGlobalMemSize = dev.deviceGlobalMemSize;
	deviceLocalMemType = dev.deviceLocalMemType;
	deviceLocalMemSize = dev.deviceLocalMemSize;
	deviceHostUnifiedMem = dev.deviceHostUnifiedMem;
	deviceMemBaseAddrAlign	= dev.deviceMemBaseAddrAlign;
	deviceName = dev.deviceName;
  context_ = dev.context_;
	programs	= dev.programs;

  for( int i = 0; i < OPENCL_NUM_INPUT_QUEUES; i++ ) {
    inputQueues[i] = dev.inputQueues[i];
  }
  currentInputQueueIdx  = dev.currentInputQueueIdx;

  for( int i = 0; i < OPENCL_NUM_COMMAND_QUEUES; i++ ) {
    commandQueues[i] = dev.commandQueues[i];
  }
  currentCommandQueueIdx  = dev.currentCommandQueueIdx;

  for( int i = 0; i < OPENCL_NUM_OUTPUT_QUEUES; i++ ) {
    outputQueues[i] = dev.commandQueues[i];
  }
  currentOutputQueueIdx   = dev.currentOutputQueueIdx;
}

OpenCLDevice::~OpenCLDevice() {

  for( int i = 0; i < OPENCL_NUM_INPUT_QUEUES; i++ ) {
    if (inputQueues[i] != NULL) {
      clReleaseCommandQueue(inputQueues[i]);
      LOG(INFO)<< "release OpenCL input queue["<<i<<"] on device " << name();
    }
  }

  for( int i = 0; i < OPENCL_NUM_COMMAND_QUEUES; i++ ) {
    if (commandQueues[i] != NULL) {
      clReleaseCommandQueue(commandQueues[i]);
      LOG(INFO)<< "release OpenCL command queue["<<i<<"] on device " << name();
    }
  }

  for( int i = 0; i < OPENCL_NUM_OUTPUT_QUEUES; i++ ) {
    if (outputQueues[i] != NULL) {
      clReleaseCommandQueue(outputQueues[i]);
      LOG(INFO)<< "release OpenCL output queue["<<i<<"] on device " << name();
    }
  }

	std::vector<cl_program>::iterator it;
	for ( it = programs.begin(); it != programs.end(); it++ ) {
		if ( *it != NULL ) {
			clReleaseProgram(*it);
			DLOG(INFO) << "release OpenCL program on platform " << name();
		}
	}
}

bool OpenCLDevice::query() {

	size_t size;

	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_TYPE, sizeof(cl_device_type), &(this->deviceType), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &(this->deviceMaxComputeUnits), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &(this->deviceMaxWorkItemDims), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, &(this->deviceWorkItemSizes[0]), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &(this->deviceMaxWorkGroupSize), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &(this->deviceMaxClockFreqMHz), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &(this->deviceMaxMemAllocSize), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t), &(this->deviceMaxArgListSize), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &(this->deviceMemAddrAlign), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, sizeof(cl_uint), &(this->deviceMinMemAddrAlign), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(cl_device_mem_cache_type), &(this->deviceGlobalMemCacheType), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &(this->deviceGlobalMemCacheSize), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint), &(this->deviceGlobalMemCacheLineSize), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &(this->deviceGlobalMemSize), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_device_local_mem_type), &(this->deviceLocalMemType), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &(this->deviceLocalMemSize), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &(this->deviceHostUnifiedMem), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &(this->deviceHostUnifiedMem), &size))) {
		return false;
	}
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &(this->deviceMemBaseAddrAlign), &size))) {
		return false;
	}
	size = 1024;
	char* name = (char*) malloc(size * sizeof(char));
	if (!CL_CHECK(clGetDeviceInfo(this->deviceID, CL_DEVICE_NAME, size, name, NULL))) {
		return false;
	}
	deviceName = std::string(name);

	switch (this->deviceType) {
	case CL_DEVICE_TYPE_CPU:
		deviceTypeStr = "CPU";
		break;
	case CL_DEVICE_TYPE_GPU:
		deviceTypeStr = "GPU";
		break;
	case CL_DEVICE_TYPE_DEFAULT:
		deviceTypeStr = "DEFAULT";
		break;
	case CL_DEVICE_TYPE_ACCELERATOR:
		deviceTypeStr = "ACCELERATOR";
		break;
#ifdef OPENCL_VERSION_1_2
	case CL_DEVICE_TYPE_CUSTOM:
		deviceTypeStr = "CUSTOM";
		break;
#endif
	case CL_DEVICE_TYPE_ALL:
		deviceTypeStr = "ALL";
		break;
	default:
		deviceTypeStr = "unknown";
	}

	return true;
}

void OpenCLDevice::print() {

	std::cout << " -- OpenCL Device Information -------------------------------------------" << std::endl;
	std::cout << "    deviceType                    = " << this->deviceTypeStr << std::endl;
	std::cout << "    deviceName                    = " << this->deviceName << std::endl;
	std::cout << "    deviceMaxComputeUnits         = " << this->deviceMaxComputeUnits << std::endl;
	std::cout << "    deviceMaxWorkItemDims         = " << this->deviceMaxWorkItemDims << std::endl;
	std::cout << "    deviceMaxWorkGroupSize        = " << this->deviceMaxWorkGroupSize << std::endl;
	std::cout << "    deviceMaxClockFreqMHz         = " << this->deviceMaxClockFreqMHz << std::endl;
	std::cout << "    deviceMaxMemAllocSize         = " << this->deviceMaxMemAllocSize << std::endl;
	std::cout << "    deviceMaxArgListSize          = " << this->deviceMaxArgListSize << std::endl;
	std::cout << "    deviceMemAddrAlign            = " << this->deviceMemAddrAlign << std::endl;
	std::cout << "    deviceMinMemAddrAlign         = " << this->deviceMinMemAddrAlign << std::endl;
	std::cout << "    deviceGlobalMemCacheType      = " << this->deviceGlobalMemCacheType << std::endl;
	std::cout << "    deviceGlobalMemCacheSize      = " << this->deviceGlobalMemCacheSize << std::endl;
	std::cout << "    deviceGlobalMemCacheLineSize  = " << this->deviceGlobalMemCacheLineSize << std::endl;
	std::cout << "    deviceGlobalMemSize           = " << this->deviceGlobalMemSize << std::endl;
	std::cout << "    deviceLocalMemType            = " << this->deviceLocalMemType << std::endl;
	std::cout << "    deviceLocalMemSize            = " << this->deviceLocalMemSize << std::endl;
	std::cout << "    deviceHostUnifiedMem          = " << this->deviceHostUnifiedMem << std::endl;
	std::cout << "    deviceMemBaseAddrAlign        = " << this->deviceMemBaseAddrAlign << std::endl;
}

cl_device_type OpenCLDevice::type() {
	return deviceType;
}

std::string OpenCLDevice::name() {
	return this->deviceName;
}

cl_device_id OpenCLDevice::id() {
	return deviceID;
}

bool OpenCLDevice::compile(std::string cl_source) {
  if (context_() == NULL) {
		LOG(ERROR)<< "cannot create OpenCL program without OpenCL context";
		return false;
	}

	if ( access( cl_source.c_str(), F_OK ) == -1 ) {
		LOG(ERROR) << "kernel source file = '" << cl_source.c_str() << "' doesn't exist";
		return false;
	}

	if ( access( cl_source.c_str(), R_OK ) == -1 ) {
		LOG(ERROR) << "kernel source file = '" << cl_source.c_str() << "' isn't readable";
		return false;
	}

	caffe::OpenCLParser parser;

	boost::regex re("\\.cl", boost::regex::perl);
	std::string cl_standard = boost::regex_replace(cl_source, re, "-STD.cl");

	if ( ! parser.convert(cl_source, cl_standard) ) {
		LOG(ERROR) << "failed to convert kernel source file = '" << cl_source.c_str() << "' to standard OpenCL";
		return false;
	}

	std::vector<std::string> kernel_names;
	if ( ! parser.getKernelNames(cl_standard, kernel_names) ) {
		LOG(ERROR) << "failed to parse kernel names from file '"<<cl_standard.c_str()<<"'";
		return false;
	}

  std::string str(std::istreambuf_iterator<char>(std::ifstream(cl_standard.c_str()).rdbuf()), std::istreambuf_iterator<char>());
	if ( str.size() <= 0 ) {
		LOG(ERROR) << "failed to read data from file = '"<< cl_standard.c_str() <<"'";
		return false;
	}

	cl_int err;
	const char *list = str.c_str();
	size_t sourceSize[] = {strlen(str.c_str())};
  cl_program program = clCreateProgramWithSource(context_(), 1, &list, sourceSize, &err);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "failed to create program from file = '"<< cl_standard.c_str() <<"'";
		return false;
	}
	DLOG(INFO) << "create program with source from file = '"<< cl_standard.c_str() <<"' for device "<<this->name();

	//-x clc++ -O5 -cl-std=CL2.0
	std::string clIncludes = "-cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math -cl-single-precision-constant -cl-denorms-are-zero -cl-mad-enable -cl-no-signed-zeros -I ../../CL/include/ -I include/caffe/util/OpenCL/";

	err = clBuildProgram(program, 1, &deviceID, clIncludes.c_str(), NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "failed to build OpenCL program from file '" << cl_standard.c_str() << "' : error = " << err;

		char* logBuffer = (char*) malloc(1024 * 1024);
		size_t tempSize;

		err = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 1000000, logBuffer, &tempSize);
		if ( err != CL_SUCCESS ) {
			LOG(ERROR) << "clGetProgramBuildInfo() failed.";
			return false;
		}
		LOG(ERROR) << this->name() << "> build log size is " << tempSize;
		LOG(ERROR) << logBuffer;

		std::ostringstream os;
		os<<"CLBuildLog_"<<this->name().c_str()<<".log";

		FILE* tempLogFile = fopen(os.str().c_str(), "w");
		if ( tempLogFile == NULL ) {
			LOG(ERROR) << "failed to open build log file '" << os.str().c_str() << "'";
		} else {
			fwrite(logBuffer, 1, tempSize, tempLogFile);
			fclose(tempLogFile);
			DLOG(INFO) << "OpenCL build log written to file '" << os.str().c_str() << "'";
		}

		return false;
	}
	DLOG(INFO) << "create program from file = '"<< cl_standard.c_str() <<"' for device " << this->name();
	programs.push_back(program);

	std::vector<std::string>::iterator it;

	for( it = kernel_names.begin(); it != kernel_names.end(); it++ ) {
		cl_kernel kern = clCreateKernel(program, (*it).c_str(), &err);
		if ( err != CL_SUCCESS ) {
			LOG(ERROR) << "failed to create kernel '"<<(*it).c_str()<<"' from file '" << cl_standard.c_str() << "' : error = " << caffe::OpenCL::what(err);
			return false;
		}

    if ( kernel_map_.find((*it)) != kernel_map_.end() ) {
			LOG(ERROR) << "kernel '" << *it << "' already present in map.";
			return false;
		}

    kernel_map_[(*it)] = kern;
    DLOG(INFO) << "create kernel '"<<(*it).c_str()<<"' from file '" << cl_standard.c_str() << "' for device " << this->name() << " @ " << kernel_map_[(*it)];
	}


	return true;
}

bool OpenCLDevice::createQueue() {

  if (!context_()) {
		LOG(ERROR)<< "cannot create command queue without context.";
		return false;
	}

  cl_int err;
	for( int i = 0; i < OPENCL_NUM_INPUT_QUEUES; i++ ) {
	  inputQueues[i] = clCreateCommandQueue(context_(), deviceID, 0, &err);
	  if ( err != CL_SUCCESS ) {
	    LOG(ERROR) << "failed to create OpenCL input queue["<<i<<"] for device " << this->name();
	    return false;
	  }
	  LOG(INFO) << "create OpenCL input queue["<<i<<"] for device " << this->name() << " @ queues = "<<inputQueues[i]<<" and deviceID = "<<deviceID;
	}

  for( int i = 0; i < OPENCL_NUM_COMMAND_QUEUES; i++ ) {
    commandQueues[i] = clCreateCommandQueue(context_(), deviceID, 0, &err);
    if ( err != CL_SUCCESS ) {
      LOG(ERROR) << "failed to create OpenCL command queue["<<i<<"] for device " << this->name();
      return false;
    }
    LOG(INFO) << "create OpenCL command queue["<<i<<"] for device " << this->name() << " @ queues = "<<commandQueues[i]<<" and deviceID = "<<deviceID;
  }

  for( int i = 0; i < OPENCL_NUM_OUTPUT_QUEUES; i++ ) {
    outputQueues[i] = clCreateCommandQueue(context_(), deviceID, 0, &err);
    if ( err != CL_SUCCESS ) {
      LOG(ERROR) << "failed to create OpenCL output queue["<<i<<"] for device " << this->name();
      return false;
    }
    LOG(INFO) << "create OpenCL output queue["<<i<<"] for device " << this->name() << " @ queues = "<<outputQueues[i]<<" and deviceID = "<<deviceID;
  }

	return true;
}

cl_context OpenCLDevice::getContext() {

  return context_();
}

cl_command_queue* OpenCLDevice::getQueue() {
  DLOG(INFO)<<"using OpenCL default command queue = 0";
	return &commandQueues[0];
}

cl_command_queue* OpenCLDevice::getNextInputQueue() {

  if ( currentInputQueueIdx < OPENCL_NUM_INPUT_QUEUES-1 ) {
    currentInputQueueIdx++;
  } else {
    currentInputQueueIdx = 0;
  }
  DLOG(INFO)<<"next OpenCL input queue = "<<currentInputQueueIdx;
  return &inputQueues[currentInputQueueIdx];
}

cl_command_queue* OpenCLDevice::getNextCommandQueue() {

  if ( currentCommandQueueIdx < OPENCL_NUM_COMMAND_QUEUES-1 ) {
    currentCommandQueueIdx++;
  } else {
    currentCommandQueueIdx = 0;
  }
  DLOG(INFO)<<"really next OpenCL command queue = "<<currentCommandQueueIdx;
  return &commandQueues[currentCommandQueueIdx];
}

cl_command_queue* OpenCLDevice::getNextOutputQueue() {

  if ( currentOutputQueueIdx < OPENCL_NUM_OUTPUT_QUEUES-1 ) {
    currentOutputQueueIdx++;
  } else {
    currentOutputQueueIdx = 0;
  }
  DLOG(INFO)<<"next OpenCL output queue = "<<currentOutputQueueIdx;
  return &outputQueues[currentOutputQueueIdx];
}

cl_command_queue* OpenCLDevice::getCurrentInputQueue() {

  DLOG(INFO)<<"current input queue = "<<currentInputQueueIdx;
  return &inputQueues[currentInputQueueIdx];
}

cl_command_queue* OpenCLDevice::getCurrentCommandQueue() {

  DLOG(INFO)<<"current command queue = "<<currentCommandQueueIdx;
  return &commandQueues[currentCommandQueueIdx];
}

cl_command_queue* OpenCLDevice::getCurrentOutputQueue() {

  DLOG(INFO)<<"current output queue = "<<currentOutputQueueIdx;
  return &outputQueues[currentOutputQueueIdx];
}

bool OpenCLDevice::setInputQueueIDX(unsigned int idx) {

  if ( idx >= OPENCL_NUM_INPUT_QUEUES ) {
    LOG(ERROR)<<"OpenCL input queue idx = "<<idx<<" out of range.";
    return false;
  }
  currentInputQueueIdx = idx;
  DLOG(INFO)<<"set OpenCL input queue idx = "<<idx;
  return true;
}

bool OpenCLDevice::setCommandQueueIDX(unsigned int idx) {

  if ( idx >= OPENCL_NUM_COMMAND_QUEUES ) {
    LOG(ERROR)<<"OpenCL command queue idx = "<<idx<<" out of range.";
    return false;
  }
  currentCommandQueueIdx = idx;
  DLOG(INFO)<<"set OpenCL command queue idx = "<<idx;
  return true;
}

bool OpenCLDevice::setOutputQueueIDX(unsigned int idx) {

  if ( idx >= OPENCL_NUM_OUTPUT_QUEUES ) {
    LOG(ERROR)<<"OpenCL output queue idx = "<<idx<<" out of range.";
    return false;
  }
  currentOutputQueueIdx = idx;
  DLOG(INFO)<<"set OpenCL output queue idx = "<<idx;
  return true;
}

void OpenCLDevice::SetContext(cl::Context context) {
  context_ = context;
}

cl_kernel* OpenCLDevice::getKernel(std::string name) {

  if (kernel_map_.find(name) == kernel_map_.end()) {
    LOG(FATAL) << "kernel '" << name << "' not found in map";
		return NULL;
	}

  return &kernel_map_[name];
}

bool OpenCLDevice::add(OpenCLMemory& clMem) {

	if ( clMem.getVirtualPointer() == NULL ) {
		LOG(ERROR) << this->name() <<" cannot add NULL pointer.";
		return false;
	}

	if ( memory.find(clMem.getVirtualPointer()) != memory.end() ) {
		LOG(ERROR) << this->name() <<" memory already exists : " << clMem.getTag().c_str();
		return false;
	}

	std::map<const void*, OpenCLMemory>::iterator it;
	for ( it = memory.begin(); it != memory.end(); it++ ) {
		if ( (it->second).overlaps(clMem) ) {
			LOG(FATAL) << "memory overlap found";
		}
	}

	DLOG(INFO) << "add memory " << clMem.getTag().c_str();
	memory[clMem.getVirtualPointer()] = clMem;

	return true;
}

bool OpenCLDevice::getBuffer(std::tr1::shared_ptr<OpenCLBuffer>* ptr, size_t size) {

  if ( size < 1 ) {
    LOG(ERROR)<<"buffer cannot be smaller than 1Byte.";
    return false;
  }

  if ( buffer.find(size) == buffer.end() ) {
    DLOG(INFO) << this->name() <<" no buffer of size = "<<size<<" found, create new buffer.";

    *ptr = std::tr1::shared_ptr<OpenCLBuffer>(new OpenCLBuffer(size));
    buffer[size].push_back(*ptr);
    this->add(**ptr);

    return true;
  }

  std::vector< std::tr1::shared_ptr<caffe::OpenCLBuffer>>::iterator it;
  for( it = buffer[size].begin(); it != buffer[size].end(); it++ ) {
    if ( (*it)->isAvailable() == true ) {
      DLOG(INFO) << this->name() <<" found available buffer of size = "<<size;
      (*it)->setUnavailable();
      *ptr = (*it);
      return true;
    }
  }

  DLOG(INFO) << this->name() <<" found no available buffer of size = "<<size<<" found, create new buffer.";
  *ptr = std::tr1::shared_ptr<OpenCLBuffer>(new OpenCLBuffer(size));
  buffer[size].push_back(*ptr);
  this->add(**ptr);
  return true;
}

bool OpenCLDevice::setBufferAvailable(const void* ptr, size_t size) {

  if ( size < 1 ) {
    LOG(ERROR)<<"buffer cannot be smaller than 1Byte.";
    return false;
  }

  if ( buffer.find(size) == buffer.end() ) {
    DLOG(ERROR) << this->name() <<" no buffer of size = "<<size<<" exists.";
    return false;
  }

  std::vector< std::tr1::shared_ptr<caffe::OpenCLBuffer>>::iterator it;
  for( it = buffer[size].begin(); it != buffer[size].end(); it++ ) {
    if ( (*it)->getVirtualPointer() == ptr ) {
      DLOG(INFO) << this->name() <<" set buffer of size = "<<size<<" with virtual address = "<<ptr<<" to available";
      (*it)->setAvailable();
      return true;
    }
  }

  DLOG(ERROR) << this->name() <<" found no buffer of size = "<<size<<" with virtual address = "<<ptr;
  return false;
}

bool OpenCLDevice::rmMemoryPtr(const void* ptr) {

	if ( ! isValidPtr(ptr) ) {
		LOG(ERROR) << this->name() <<" not a valid pointer @ " << ptr <<" on this device.";
		return false;
	}

	if ( memory.find(ptr) == memory.end() ) {
		LOG(ERROR) << this->name() <<" not a valid key @ " << ptr <<" on this device.";
		return false;
	}
	try {
		memory[ptr].free();
	} catch (std::exception& e) {
		e.what();
		return false;
	}

	DLOG(INFO)<<"free "<<memory[ptr].getTag();

	int count = memory.erase(ptr);
	DLOG(INFO) << count << " erased for " << ptr;

	if ( memory.find(ptr) != memory.end() ) {
		LOG(ERROR) << this->name() <<" ptr @ " << ptr <<" still on this device.";
		return false;
	}

	return true;
}

bool OpenCLDevice::isValidPtr(const void* ptr) {

	if ( ptr == NULL ) {
		return false;
	}

	if ( ! caffe::OpenCLMemory::isHighMem(ptr) ) {
		return false;
	}

	return true;
	/*
	// direct match of the base address
	if ( memory.find(ptr) != memory.end() ) {
		return true;
	}

	std::map<const void*, OpenCLMemory>::iterator it;
	for ( it = memory.begin(); it != memory.end(); it++ ) {
		if ( (it->second).contains(ptr) ) {
			return true;
		}
	}
	return false;
	*/
}

bool OpenCLDevice::get(const void* ptr, OpenCLMemory** clMem) {

  if ( ptr == NULL ) {
    return false;
  }

  if ( ! caffe::OpenCLMemory::isHighMem(ptr) ) {
    LOG(ERROR)<<"memory pointer to query is not in virtual memory.";
    return false;
  }

  // direct match of the base address
  if ( memory.find(ptr) != memory.end() ) {
    *clMem = &(memory[ptr]);
    return true;
  }

  int found = 0;
  std::map<const void*, OpenCLMemory>::iterator it;
  for ( it = memory.begin(); it != memory.end(); it++ ) {
    if ( (it->second).contains(ptr) ) {
      found++;
    }
  }

  if ( found > 1 ) {
    DLOG(INFO) << "found "<<found<<" matching for "<<ptr;
    for ( it = memory.begin(); it != memory.end(); it++ ) {
      if ( (it->second).contains(ptr) ) {
        DLOG(INFO) << it->second.getTag().c_str();
      }
    }

  }

  for ( it = memory.begin(); it != memory.end(); it++ ) {
    if ( (it->second).contains(ptr) ) {
      *clMem = &(it->second);
      return true;
    }
  }

  return false;
}

std::string OpenCLDevice::getMemoryTag(const void* ptr) {

	if ( ptr == NULL ) {
		return "NULL";
	}

	if ( ! caffe::OpenCLMemory::isHighMem(ptr) ) {
		return "NO VIRTUAL ADDRESS";
	}

	OpenCLMemory* clMem;
	if ( ! get(ptr, &clMem) ) {
		return "UNKNOWN";
	}

	return clMem->getTag();
}

std::string OpenCLDevice::getDeviceName() {

	return deviceName;
}

cl_uint OpenCLDevice::getDeviceMemBaseAddrAlign() {

	return deviceMemBaseAddrAlign;
}

size_t OpenCLDevice::getMemoryUsage() {

	size_t bytesUsed = 0;
	std::map<const void*, OpenCLMemory>::iterator it;
	for ( it = memory.begin(); it != memory.end(); it++ ) {
		OpenCLMemory* clMem = &(it->second);
		bytesUsed += clMem->getSize();
	}
	return bytesUsed;
}

void OpenCLDevice::waitForInputQueues() {

  for( int i = 0; i < OPENCL_NUM_INPUT_QUEUES; i++ ) {
    DLOG(INFO)<<"calling clFinish(inputQueues["<<i<<"])";
    CL_CHECK(clFinish(inputQueues[i]));
  }
}

void OpenCLDevice::waitForCommandQueues() {

  for( int i = 0; i < OPENCL_NUM_COMMAND_QUEUES; i++ ) {
    DLOG(INFO)<<"calling clFinish(commandQueues["<<i<<"])";
    CL_CHECK(clFinish(commandQueues[i]));
  }
}

void OpenCLDevice::waitForOutputQueues() {

  for( int i = 0; i < OPENCL_NUM_OUTPUT_QUEUES; i++ ) {
    DLOG(INFO)<<"calling clFinish(outputQueues["<<i<<"])";
    CL_CHECK(clFinish(outputQueues[i]));
  }
}

void OpenCLDevice::Synchronize() {

  for( int i = 0; i < OPENCL_NUM_INPUT_QUEUES; i++ ) {
    DLOG(INFO)<<"calling clFinish(inputQueues["<<i<<"])";
    CL_CHECK(clFinish(inputQueues[i]));
  }
  for( int i = 0; i < OPENCL_NUM_COMMAND_QUEUES; i++ ) {
    DLOG(INFO)<<"calling clFinish(commandQueues["<<i<<"])";
    CL_CHECK(clFinish(commandQueues[i]));
  }
  for( int i = 0; i < OPENCL_NUM_OUTPUT_QUEUES; i++ ) {
    DLOG(INFO)<<"calling clFinish(outputQueues["<<i<<"])";
    CL_CHECK(clFinish(outputQueues[i]));
  }
}

} // namespace caffe

#endif //USE_OPENCL
