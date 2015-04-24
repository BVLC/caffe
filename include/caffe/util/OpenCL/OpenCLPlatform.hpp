#ifndef __OPENCL_PLATFORM_HPP__
#define __OPENCL_PLATFORM_HPP__

#include <CL/cl.hpp>
#include <vector>
#include <caffe/util/OpenCL/OpenCLDevice.hpp>

namespace caffe {

class OpenCLPlatform {
public:
	OpenCLPlatform();
	OpenCLPlatform(const OpenCLPlatform& pf);
  OpenCLPlatform(cl::Platform platform);

	~OpenCLPlatform();

  bool Query();
	void print();
	int  getNumCPUDevices();
	int  getNumGPUDevices();
  std::string name();
  std::string vendor();
  std::string version();
  std::string profile();
  void SetCurrentDevice(int device_index);
  OpenCLDevice& CurrentDevice();
	OpenCLDevice* getDevice(cl_device_type type, unsigned int idx);
	cl_platform_id id();
	bool createContext();
	bool compile(std::string sources);
	int getNumDevices(cl_device_type type);
private:
  void Check();
  //cl_platform_id platformID;
  cl::Platform platform_;
  std::string name_;
  std::string vendor_;
  std::string version_;
  std::string extensions_;
  std::string profile_;
	cl_uint numCPUDevices;
	cl_uint numGPUDevices;
	cl_uint numDevices;
	cl_device_id*	devicePtr;
	cl_device_id*	cpuDevicePtr;
	cl_device_id*	gpuDevicePtr;
	std::vector<caffe::OpenCLDevice>	devices;
//  cl_context context;
  cl::Context context_;
//	std::vector<cl_program> programs;
  int current_device_index_;
};

} // namespace caffe

#endif // __OPENCL_PLATFORM_HPP__
