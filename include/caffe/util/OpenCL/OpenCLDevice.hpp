#ifndef __OPENCL_DEVICE_HPP__
#define __OPENCL_DEVICE_HPP__

#include <CL/cl.h>
#include <CL/cl.hpp>
#include <string>
#include <iostream>
#include <map>
#include <set>
#include <caffe/util/OpenCL/OpenCLMemory.hpp>

namespace caffe {

class OpenCLDevice {

public:
	OpenCLDevice();
  OpenCLDevice(cl_platform_id pid, cl_device_id did);
	OpenCLDevice(const OpenCLDevice& dev);
	~OpenCLDevice();

	bool query();
	void print();
  void SetContext(cl::Context context);
	cl_device_type type();
	std::string name();
	cl_device_id id();
  //bool createContext();
	bool compile(std::string source);
	bool createQueue();
  cl_context getContext();
	cl_command_queue* getQueue();
	cl_kernel* getKernel(std::string name);
	bool add(OpenCLMemory& clMem);
	bool rmMemoryPtr(const void* ptr);
	bool isValidPtr(const void* ptr);
	std::string getMemoryTag(const void* ptr);
	bool get(const void* ptr, OpenCLMemory& clMem);
	std::string getDeviceName();
	cl_uint getDeviceMemBaseAddrAlign();
	size_t getMemoryUsage();
  void Synchronize();
protected:

private:

	cl_device_id deviceID;
	cl_platform_id platformID;
	cl_device_type deviceType;
	std::string deviceTypeStr;
	cl_uint deviceMaxComputeUnits;
	cl_uint deviceMaxWorkItemDims;
	size_t deviceWorkItemSizes[3];
	size_t deviceMaxWorkGroupSize;
	cl_uint deviceMaxClockFreqMHz;
	cl_ulong deviceMaxMemAllocSize;
	size_t deviceMaxArgListSize;
	cl_uint deviceMemAddrAlign;
	cl_uint deviceMinMemAddrAlign;
	cl_device_mem_cache_type deviceGlobalMemCacheType;
	cl_ulong deviceGlobalMemCacheSize;
	cl_uint deviceGlobalMemCacheLineSize;
	cl_ulong deviceGlobalMemSize;
	cl_device_local_mem_type deviceLocalMemType;
	cl_ulong deviceLocalMemSize;
	cl_bool deviceHostUnifiedMem;
	cl_uint deviceMemBaseAddrAlign;
	std::string deviceName;
  cl::Context context_;
	std::vector<cl_program> programs;
	cl_command_queue queue;
  std::map<std::string, cl_kernel> kernel_map_;

	std::map<const void*, caffe::OpenCLMemory> memory;
};

} // namespace caffe

#endif // __OPENCL_DEVICE_HPP__
