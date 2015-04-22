#ifdef USE_OPENCL

#include <glog/logging.h>
#include <CL/cl.h>
#include <caffe/util/OpenCL/OpenCLPlatform.hpp>
#include <caffe/util/OpenCL/OpenCLSupport.hpp>
#include <caffe/util/OpenCL/OpenCLParser.hpp>
#include "unistd.h"
#include "stdio.h"
#include <iostream>
#include <cstring>
#include <fstream>

namespace caffe {

OpenCLPlatform::OpenCLPlatform():
  platform_(),
  current_device_index_(-1) {
	numCPUDevices 		= 0;
	numGPUDevices 		= 0;
	numDevices	  		= 0;
	cpuDevicePtr 		= NULL;
	gpuDevicePtr 		= NULL;
	devicePtr	 		= NULL;
	context				= NULL;
}

OpenCLPlatform::OpenCLPlatform(const OpenCLPlatform& pf):
  name_(pf.name_),
  vendor_(pf.vendor_),
  version_(pf.version_),
  extensions_(pf.extensions_),
  profile_(pf.profile_),
  current_device_index_(-1) {
  numCPUDevices 		= pf.numCPUDevices;
	numGPUDevices 		= pf.numGPUDevices;
	numDevices	  		= pf.numDevices;
	cpuDevicePtr 		= pf.cpuDevicePtr;
	gpuDevicePtr 		= pf.gpuDevicePtr;
	devicePtr	 		= pf.devicePtr;
	devices				= pf.devices;
	context				= pf.context;
	programs			= pf.programs;
}

OpenCLPlatform::OpenCLPlatform(cl::Platform platform):
  platform_(platform),
  current_device_index_(-1) {
	numCPUDevices 		= 0;
	numGPUDevices 		= 0;
	numDevices	  		= 0;
	cpuDevicePtr 		= NULL;
	gpuDevicePtr 		= NULL;
	devicePtr	 		= NULL;
	context 			= NULL;
}

OpenCLPlatform::~OpenCLPlatform() {
	if ( context != NULL ) {
		clReleaseContext(context);
		LOG(INFO) << "release OpenCL context on platform " << name();
	}

	std::vector<cl_program>::iterator it;
	for ( it = programs.begin(); it != programs.end(); it++ ) {
		if ( (*it) != NULL ) {
			//clReleaseProgram((*it));
			//LOG(INFO) << "release OpenCL program on platform " << name();
		}
	}
}

// Check that we have an assigned cl::Platform object.
void OpenCLPlatform::Check() {
  if (!platform_()) {
    LOG(FATAL) << "Unassigned platform.";
  }
}

bool OpenCLPlatform::Query() {
  Check();

	// get the name
  cl_int err;
  name_ = platform_.getInfo<CL_PLATFORM_NAME>(&err);
  if (!CL_CHECK(err)) return false;

	// get the vendor
  vendor_ = platform_.getInfo<CL_PLATFORM_VENDOR>(&err);
  if (!CL_CHECK(err)) return false;

  version_ = platform_.getInfo<CL_PLATFORM_VERSION>(&err);
  if (!CL_CHECK(err)) return false;

  extensions_ = platform_.getInfo<CL_PLATFORM_EXTENSIONS>(&err);
  if (!CL_CHECK(err)) return false;

  profile_ = platform_.getInfo<CL_PLATFORM_PROFILE>(&err);
  if (!CL_CHECK(err)) return false;

	// CPU & GPU devices
  if (!CL_CHECK( clGetDeviceIDs(platform_(), CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, 0, NULL, &(numDevices)) ) ) {
		return false;
	}
	devicePtr = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
  if (!CL_CHECK( clGetDeviceIDs(platform_(), CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, numDevices, devicePtr, NULL) ) ) {
		return false;
	}

	// CPU devices
  if ( clGetDeviceIDs(platform_(), CL_DEVICE_TYPE_CPU, 0, NULL, &(numCPUDevices)) == CL_SUCCESS ) {
		cpuDevicePtr = (cl_device_id*) malloc(numCPUDevices * sizeof(cl_device_id));
    if ( ! CL_CHECK( clGetDeviceIDs(platform_(), CL_DEVICE_TYPE_CPU, numCPUDevices, cpuDevicePtr, NULL) ) ) {
			return false;
		}

		for (int i = 0; i < numCPUDevices; i++) {
      OpenCLDevice d = OpenCLDevice(platform_(), cpuDevicePtr[i]);
			d.query();
			devices.push_back(d);
		}
	}

	// GPU DEVICES
  if (!CL_CHECK( clGetDeviceIDs(platform_(), CL_DEVICE_TYPE_GPU, 0, NULL, &(numGPUDevices)) ) ) {
		return false;
	}
	gpuDevicePtr = (cl_device_id*) malloc(numGPUDevices * sizeof(cl_device_id));
  if ( ! CL_CHECK( clGetDeviceIDs(platform_(), CL_DEVICE_TYPE_GPU, numGPUDevices, gpuDevicePtr, NULL) ) ) {
		return false;
	}

	for (int i = 0; i < numGPUDevices; i++) {
    OpenCLDevice d = OpenCLDevice(platform_(), gpuDevicePtr[i]);
		d.query();
		devices.push_back(d);
	}
	return true;
}

void OpenCLPlatform::print() {
  std::cout << "  Platform Name       " <<  name_ << std::endl;
  std::cout << "  Platform Vendor     " <<  vendor_ << std::endl;
  std::cout << "  Platform Version    " <<  version_ << std::endl;
  std::cout << "  Platform Extensions " <<  extensions_ << std::endl;
  std::cout << "  Platform Profile    " <<  profile_ << std::endl;
  std::cout << "  Number of CPU Devs  " <<  numCPUDevices << std::endl;
  std::cout << "  Number of GPU Devs  " <<  numGPUDevices << std::endl;

	std::vector<caffe::OpenCLDevice>::iterator it;
	for (it = devices.begin(); it != devices.end(); it++) {
		(*it).print();
	}
}

int OpenCLPlatform::getNumCPUDevices() {
	return numCPUDevices;
}

int OpenCLPlatform::getNumGPUDevices() {
	return numGPUDevices;
}

std::string OpenCLPlatform::name() {
  return name_;
}

std::string OpenCLPlatform::vendor() {
  return vendor_;
}

std::string OpenCLPlatform::version() {
  return version_;
}

std::string OpenCLPlatform::profile() {
  return profile_;
}

OpenCLDevice* OpenCLPlatform::getDevice(cl_device_type type, unsigned int idx) {
	switch(type) {
	case CL_DEVICE_TYPE_CPU:
		if ( idx >= numCPUDevices ) {
			LOG(ERROR) << "CPU device index out of range";
			return NULL;
		}
		break;
	case CL_DEVICE_TYPE_GPU:
		if ( idx >= numGPUDevices ) {
			LOG(ERROR) << "GPU device index out of range";
			return NULL;
		}
		break;
	default:
		LOG(ERROR) << "device type unsupported.";
		return NULL;
	}

	std::vector<caffe::OpenCLDevice>::iterator it;
	int dev_cnt = 0;
	int gpu_cnt = 0;
	int cpu_cnt = 0;
	for (it = devices.begin(); it != devices.end(); it++) {
		if ( (*it).type() == CL_DEVICE_TYPE_CPU && (*it).type() == type ) {
			if ( cpu_cnt == idx ) {
				return &devices[dev_cnt];
			}
			cpu_cnt++;
		}
		if ( (*it).type() == CL_DEVICE_TYPE_GPU && (*it).type() == type ) {
			if ( gpu_cnt == idx ) {
				return &devices[dev_cnt];
			}
			gpu_cnt++;
		}
		dev_cnt++;
	}

	return NULL;
}

int OpenCLPlatform::getNumDevices(cl_device_type type) {

	switch(type) {
	case CL_DEVICE_TYPE_CPU:
		return numCPUDevices;
		break;
	case CL_DEVICE_TYPE_GPU:
		return numGPUDevices;
		break;
	default:
		LOG(ERROR) << "device type unsupported.";
		return -1;
	}
}

cl_platform_id OpenCLPlatform::id() {
  return platform_();
}

bool OpenCLPlatform::createContext() {
	cl_int err;
  cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform_(), 0};
	context = clCreateContext(contextProperties, numDevices, devicePtr, NULL, NULL, &err);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "failed to create context.";
		return false;
	}
	LOG(INFO) << "create OpenCL context for platform " << this->name();

	std::vector<caffe::OpenCLDevice>::iterator it;
	for (it = devices.begin(); it != devices.end(); it++) {
		if ( ! (*it).createContext() ) {
			return false;
		}
	}

	return true;
}

bool OpenCLPlatform::compile(std::string cl_source) {
	if ( context == NULL ) {
		LOG(ERROR) << "cannot create OpenCL program without OpenCL context";
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
	cl_program program = clCreateProgramWithSource(context, 1, &list, sourceSize, &err);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "failed to create program from file = '"<< cl_standard.c_str() <<"'";
		return false;
	}
	DLOG(INFO) << "create program with source from file = '"<< cl_standard.c_str() <<"' for platform " << this->name();

	//-x clc++ -O5
	std::string clIncludes = std::string("-cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math -cl-single-precision-constant -cl-denorms-are-zero -cl-mad-enable -cl-no-signed-zeros -I ../../CL/include/");
	err = clBuildProgram(program, numDevices, devicePtr, clIncludes.c_str(), NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "failed to build OpenCL program from file '" << cl_standard.c_str() << "' : error = " << err;

		char* logBuffer = (char*) malloc(1024 * 1024);
		size_t tempSize;

		std::vector<caffe::OpenCLDevice>::iterator it;
		for (it = devices.begin(); it != devices.end(); it++) {
			err = clGetProgramBuildInfo(program, (*it).id(), CL_PROGRAM_BUILD_LOG, 1000000, logBuffer, &tempSize);
			if ( err != CL_SUCCESS ) {
				LOG(ERROR) << "clGetProgramBuildInfo() failed.";
				return false;
			}
			LOG(ERROR) << (*it).name() << "> build log size is " << tempSize;
			LOG(ERROR) << logBuffer;

			std::ostringstream os;
			os<<"CLBuildLog_"<<(*it).name().c_str()<<".log";

			FILE* tempLogFile = fopen(os.str().c_str(), "w");
			if ( tempLogFile == NULL ) {
				LOG(ERROR) << "failed to open build log file '" << os.str().c_str() << "'";
			} else {
				fwrite(logBuffer, 1, tempSize, tempLogFile);
				fclose(tempLogFile);
				DLOG(INFO) << "OpenCL build log written to file '" << os.str().c_str() << "'";
			}
		}
		return false;
	}
	DLOG(INFO) << "create program for all devices from file = '"<< cl_standard.c_str() <<"' for platform " << this->name();
	programs.push_back(program);

	std::vector<caffe::OpenCLDevice>::iterator it;
	for (it = devices.begin(); it != devices.end(); it++) {
		if ( ! (*it).compile(cl_source) ) {
			return false;
		}
	}

	return true;
}

OpenCLDevice& OpenCLPlatform::CurrentDevice() {
  if (current_device_index_ < 0 ) {
    LOG(FATAL) << "Current device not set.";
  }
//  OpenCLDevice& device = pf.getDevice(CL_DEVICE_TYPE_GPU, 0);
//  if (!gpu) {
//    LOG(ERROR) << "failed to select first GPU on platform " << pf.name();
//		return false;
//	}

  return devices[current_device_index_];
}

void OpenCLPlatform::SetCurrentDevice(int device_index) {
  if (device_index >= numCPUDevices + numGPUDevices ||
      device_index < 0) {
    LOG(FATAL) << "Device index out of range";
  }
  current_device_index_ = device_index;
}


} // namespace caffe

#endif // USE_OPENCL
