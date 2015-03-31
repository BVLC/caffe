#ifndef __OPENCL_PLATFORM_HPP__
#define __OPENCL_PLATFORM_HPP__

#include <CL/cl.h>
#include <vector>
#include <caffe/util/OpenCL/OpenCLDevice.hpp>

namespace caffe {

class OpenCLPlatform {

public:
	OpenCLPlatform();
	OpenCLPlatform(const OpenCLPlatform& pf);
	OpenCLPlatform(cl_platform_id id);

	~OpenCLPlatform();

	bool query();
	void print();
	int  getNumCPUDevices();
	int  getNumGPUDevices();
	char* name();
	char* vendor();
	char* version();
	char* profile();
	OpenCLDevice* getDevice(cl_device_type type, unsigned int idx);
	cl_platform_id id();
	bool createContext();
	bool compile(std::string sources);
	int getNumDevices(cl_device_type type);

protected:

private:

	cl_platform_id platformID;
	char* platformName;
	char* platformVendor;
	char* platformVersion;
	char* platformExtensions;
	char* platformProfile;
	cl_uint numCPUDevices;
	cl_uint numGPUDevices;
	cl_uint numDevices;
	cl_device_id*	devicePtr;
	cl_device_id*	cpuDevicePtr;
	cl_device_id*	gpuDevicePtr;
	std::vector<caffe::OpenCLDevice>	devices;
	cl_context context;
	std::vector<cl_program> programs;
};

} // namespace caffe

#endif // __OPENCL_PLATFORM_HPP__
