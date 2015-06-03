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
}

OpenCLPlatform::OpenCLPlatform(const OpenCLPlatform& pf):
  platform_(pf.platform_),
  name_(pf.name_),
  vendor_(pf.vendor_),
  version_(pf.version_),
  extensions_(pf.extensions_),
  profile_(pf.profile_),
  current_device_index_(pf.current_device_index_) {
  numCPUDevices 		= pf.numCPUDevices;
	numGPUDevices 		= pf.numGPUDevices;
	numDevices	  		= pf.numDevices;
	cpuDevicePtr 		= pf.cpuDevicePtr;
	gpuDevicePtr 		= pf.gpuDevicePtr;
	devicePtr	 		= pf.devicePtr;
	devices				= pf.devices;
  context_				= pf.context_;
//	programs			= pf.programs;
}

OpenCLPlatform::OpenCLPlatform(cl::Platform platform) : platform_(platform), current_device_index_(-1) {
	numCPUDevices 		= 0;
	numGPUDevices 		= 0;
	numDevices	  		= 0;
	cpuDevicePtr 		= NULL;
	gpuDevicePtr 		= NULL;
	devicePtr	 		= NULL;
}

OpenCLPlatform::~OpenCLPlatform() {
//	if ( context != NULL ) {
//		clReleaseContext(context);
//		LOG(INFO) << "release OpenCL context on platform " << name();
//	}

//	std::vector<cl_program>::iterator it;
//	for ( it = programs.begin(); it != programs.end(); it++ ) {
//		if ( (*it) != NULL ) {
//			//clReleaseProgram((*it));
//			//LOG(INFO) << "release OpenCL program on platform " << name();
//		}
//	}
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
  cl_int err = 0;
  name_ = platform_.getInfo<CL_PLATFORM_NAME>(&err);
  if (!CL_CHECK(err)) { name_ = "Failed to get platform name."; }

	// get the vendor
  vendor_ = platform_.getInfo<CL_PLATFORM_VENDOR>(&err);
  if (!CL_CHECK(err)) { vendor_ = "Failed to get platform vendor."; }

  version_ = platform_.getInfo<CL_PLATFORM_VERSION>(&err);
  if (!CL_CHECK(err)) { version_ = "Failed to get platform version."; }

  extensions_ = platform_.getInfo<CL_PLATFORM_EXTENSIONS>(&err);
  if (!CL_CHECK(err)) { extensions_ = "failed to get platform extensions."; }

  profile_ = platform_.getInfo<CL_PLATFORM_PROFILE>(&err);
  if (!CL_CHECK(err)) { profile_ = "Failed to get platform profile."; }

	// CPU & GPU devices
  if (!CL_CHECK( clGetDeviceIDs(platform_(),
                                CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU,
                                0, NULL, &(numDevices)) ) ) {
		return false;
	}
	devicePtr = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
  if (!CL_CHECK( clGetDeviceIDs(platform_(),
                                CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU,
                                numDevices, devicePtr, NULL) ) ) {
		return false;
	}

	// CPU devices
  if ( clGetDeviceIDs(platform_(), CL_DEVICE_TYPE_CPU, 0,
                      NULL, &(numCPUDevices)) == CL_SUCCESS ) {
    cpuDevicePtr =
        (cl_device_id*) malloc(numCPUDevices * sizeof(cl_device_id));
    if ( ! CL_CHECK( clGetDeviceIDs(platform_(),
                     CL_DEVICE_TYPE_CPU,
                     numCPUDevices, cpuDevicePtr, NULL) ) ) {
			return false;
		}

		for (int i = 0; i < numCPUDevices; i++) {
      OpenCLDevice d(platform_(), cpuDevicePtr[i]);
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
    OpenCLDevice d(platform_(), gpuDevicePtr[i]);
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
  context_ = cl::Context(CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU,
                         contextProperties, NULL, NULL, &err);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "failed to create context.";
		return false;
	}
	LOG(INFO) << "create OpenCL context for platform " << this->name();

	std::vector<caffe::OpenCLDevice>::iterator it;
	for (it = devices.begin(); it != devices.end(); it++) {
      it->SetContext(context_);
	}

	return true;
}

bool OpenCLPlatform::compile(std::string cl_source) {
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
  int cpuDeviceCount = 0;
  int gpuDeviceCount = 0;

  switch(current_device_type_) {
    case CL_DEVICE_TYPE_CPU:
      for(int i = 0; i < devices.size(); i++) {
        if ( devices[i].type() == CL_DEVICE_TYPE_CPU ) {
          if ( cpuDeviceCount == current_device_index_ ) {
            return devices[i];
          }
          cpuDeviceCount++;
        }
      }
      break;
    case CL_DEVICE_TYPE_GPU:
      for(int i = 0; i < devices.size(); i++) {
        if ( devices[i].type() == CL_DEVICE_TYPE_GPU ) {
          if ( gpuDeviceCount == current_device_index_ ) {
            return devices[i];
          }
          gpuDeviceCount++;
        }
      }
      break;
    default:
      LOG(FATAL) << "unsupported CL_DEVICE_TYPE = "<<current_device_type_;
  }
  return devices[current_device_index_];
}

void OpenCLPlatform::DeviceSynchronize() {
  CurrentDevice().Synchronize();
}

void OpenCLPlatform::SetCurrentDevice(cl_device_type type, int device_index) {

  switch(type) {
    case CL_DEVICE_TYPE_CPU:
      if (device_index >= numCPUDevices || device_index < 0) {
        LOG(FATAL) << "Device index for CL_DEVICE_TYPE_CPU out of range";
      }
      current_device_index_ = device_index;
      current_device_type_  = type;
      break;
    case CL_DEVICE_TYPE_GPU:
      if (device_index >= numGPUDevices || device_index < 0) {
        LOG(FATAL) << "Device index for CL_DEVICE_TYPE_GPU out of range";
      }
      current_device_index_ = device_index;
      current_device_type_  = type;
      break;
    default:
      LOG(FATAL) << "unsupported CL_DEVICE_TYPE = "<<type;
  }
}

} // namespace caffe

#endif // USE_OPENCL
