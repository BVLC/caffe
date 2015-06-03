#ifdef USE_OPENCL

#ifndef __OPENCL_SUPPORT_HPP__
#define __OPENCL_SUPPORT_HPP__
#include <sstream>
#include <iostream>
#include <string>
#include <CL/cl.h>
#include "sys/time.h"
#include "sys/types.h"
#include "time.h"

#ifdef USE_OPENCL
#include <clBLAS.h>
#include <caffe/util/OpenCL/OpenCLDevice.hpp>
#include <caffe/util/OpenCL/OpenCLPlatform.hpp>
#include <caffe/util/OpenCL/OpenCLManager.hpp>
#include <caffe/util/OpenCL/OpenCLMemory.hpp>

#include <glog/logging.h>

#ifndef OPENCL_OPT_LEVEL
#define OPENCL_OPT_LEVEL 0
#endif

#ifndef CAFFE_OPENCL_VERSION
#define CAFFE_OPENCL_VERSION "0.0"
#endif

#define OPENCL_LOCAL_SIZE 64

#if defined(CL_API_SUFFIX__VERSION_2_0)
#define OPENCL_VERSION 2.0
#define OPENCL_VERSION_2_0
#define OPENCL_VERSION_1_2
#define OPENCL_VERSION_1_1
#define OPENCL_VERSION_1_0
#elif defined(CL_API_SUFFIX__VERSION_1_2)
#define OPENCL_VERSION 1.2
#define OPENCL_VERSION_1_2
#define OPENCL_VERSION_1_1
#define OPENCL_VERSION_1_0
#elif defined(CL_API_SUFFIX__VERSION_1_1)
#define OPENCL_VERSION 1.1
#define OPENCL_VERSION_1_1
#define OPENCL_VERSION_1_0
#elif defined(CL_API_SUFFIX__VERSION_1_0)
#define OPENCL_VERSION 1.0
#define OPENCL_VERSION_1_0
#else
#define OPENCL_VERSION 0.0
#endif

#define CL_PTR_BIT 63

#define CL_CHECK(code) \
	({ \
		bool ret = false;\
		std::ostringstream message;\
		message << "[" << __FILE__ << " > " << __func__ << "():" << __LINE__ << "]";\
		if ( code != CL_SUCCESS ) { \
			message << " failed: " << caffe::OpenCL::what(code) << " : " << code; \
			std::cerr << message.str() << std::endl; \
			ret = false;\
		} else { \
			ret = true;\
		}\
		ret;\
	})\

#define BOOL_CHECK(code) \
	({ \
		bool ret = false;\
		std::ostringstream message;\
		message << "[" << __FILE__ << " > " << __func__ << "():" << __LINE__ << "]";\
		if ( code != true ) { \
			message << " failed."; \
			std::cerr << message.str() << std::endl; \
			ret = false;\
		} else { \
			ret = true;\
		}\
		ret;\
	})\

#endif

#define CL_SET_KERNEL_ARG\
	cl_int err;\
	unsigned int idx = 0;\
	std::vector<cl_mem> sb;\
	std::map<const void*, std::pair<void*, size_t> > bm;

#define CL_SET_TYPE_KERNEL_ARG(type, variable, kernel) \
  if ( ! clSetKernelTypeArg(variable, idx, kernel) ) return false;

#define CL_SET_ARRAY_KERNEL_ARG(variable, kernel) \
  if ( ! clSetKernelArrayArg(*variable, idx, sb, bm, kernel) ) return false;

#define CL_SET_KERNEL_ARG_END\
	clReleaseSubBuffers(sb);\
	clReleaseBufferMap(bm);\

#define _format(message,...)  "%s[%6d] in %s :" message, __FILE__, __LINE__, __func__,  ##__VA_ARGS__

namespace caffe {

namespace OpenCL {

	int64_t getTickCount();
	double getTickFrequency();

//	bool init();

	bool clMalloc(void** virtualPtr, size_t);
	bool clFree(void* virtualPtr);
	template<typename T> bool clMemset(void* gpuPtr, const T alpha, const size_t Bytes);
	template<typename T> bool clGPU2GPU(const void* src, void* dst, size_t Bytes);
	bool clMemcpy(void* dst, const void* src, size_t Bytes, int type);
	bool clIsVirtualMemory(const void* p);
	bool clMakeLogical(const void* ptr_virtual, const void** ptr_logical);
	bool clMakeLogical2(const void* ptr_virtual, const void** ptr_logical, std::vector<cl_mem>& subBuffers, std::map<const void*, std::pair<void*, size_t> >& bufferMap);
  template<typename T> bool clSetKernelTypeArg(T variable, unsigned int& idx,
                                               cl_kernel* kernel);
  bool clSetKernelArrayArg(const void* ptr_virtual, unsigned int& idx, std::vector<cl_mem>& subBuffers, std::map<const void*, std::pair<void*, size_t> >& bufferMap, cl_kernel* kernel);
	bool clReleaseSubBuffers(std::vector<cl_mem>& subBuffers);
	bool clReleaseBufferMap(std::map<const void*, std::pair<void*, size_t> >& bufferMap);

	size_t clGetMemoryOffset(const void* ptr_virtual);
	size_t clGetMemorySize(const void* ptr_virtual);
	void* clGetMemoryBase(const void* ptr_virtual);

	template<typename T> std::string clGetKernelName(std::string name);

	template<typename T> bool clsign(const int n, const void* array_GPU_x, void* array_GPU_y);
	template<typename T> bool clsgnbit(const int n, const void* array_GPU_x, void* array_GPU_y);
	template<typename T> bool clabs(const int n, const void* array_GPU_x, void* array_GPU_y);
	template<typename T> bool cldiv(const int n, const void* array_GPU_x, const void* array_GPU_y, void* array_GPU_z);
	template<typename T> bool clmul(const int n, const void* array_GPU_x, const void* array_GPU_y, void* array_GPU_z);
	template<typename T> bool clsub(const int n, const T* array_GPU_x, const T* array_GPU_y, T* array_GPU_z);
	template<typename T> bool cladd(const int n, const T* array_GPU_x, const T* array_GPU_y, T* array_GPU_z);
	template<typename T> bool cladd_scalar(const int N, const T alpha, T* Y);
	template<typename T> bool clpowx(const int n, const T* array_GPU_x, const T alpha, T* array_GPU_z);
	template<typename T> bool clexp(const int n, const T* array_GPU_x, T* array_GPU_y);
	template<typename T> bool clgemm(const int m, const int n, const int k, const T alpha, const T* A, const T* B, const T beta, T* C);

	/* clBLAS wrapper functions */
	template<typename T> bool clBLASasum(const int n, const void* gpuPtr, T* y);
	template<typename T> bool clBLASscal(const int n, const float alpha, const void* array_GPU_x, void* array_GPU_y);
	template<typename T> bool clBLASdot(const int n, const T* x, const int incx, const T* y, const int incy, T* out);
	template<typename T> bool clBLASgemv(const clblasTranspose TransA, const int m, const int n, const T alpha, const T* A, const T* x, const T beta, T* y);
	template<typename T> bool clBLASgemv(const clblasTranspose TransA, const int m, const int n, const T alpha, const T* A, const int step_A, const T* x, const int step_x, const T beta, T* y, const int step_y);
	template<typename T> bool clBLASgemm(const clblasTranspose TransA, const clblasTranspose TransB, const int m, const int n, const int k, const T alpha, const T* A, const T* x, const T beta, T* y);
	template<typename T> bool clBLASgemm(const clblasTranspose TransA, const clblasTranspose TransB, const int m, const int n, const int k, const T alpha, const T* A, const int step_A, const T* x, const int step_x, const T beta, T* y, const int step_y);
	template<typename T> bool clBLASaxpy(const int N, const T alpha, const T* X, const int incr_x, T* Y, const int incr_y);
	bool cl_caffe_gpu_rng_uniform(const int n, unsigned int* r);
	template<typename T> bool cl_caffe_gpu_rng_uniform(const int n, const T a, const T b, T* r);
	template<typename T> bool cl_caffe_gpu_rng_gaussian(const int n, const T mu, const T sigma, T* r);
	template<typename T1, typename T2> bool cl_caffe_gpu_rng_bernoulli(const int n, const T1 p, T2* r);

	const char* what(cl_int value);

	const static int COPY_CPU_TO_CPU = 0;
	const static int COPY_CPU_TO_GPU = 1;
	const static int COPY_GPU_TO_CPU = 2;
	const static int COPY_GPU_TO_GPU = 3;
	const static int COPY_DEFAULT    = 4;

//	extern bool inititialized;
//	extern caffe::OpenCLManager mgr;
//	extern caffe::OpenCLPlatform* pf;
//	extern caffe::OpenCLDevice* gpu;
//	extern cl_context*	context;
//	extern cl_command_queue* queue;
//	extern cl_kernel* kernel;

} // namespace OpenCL

} // namespace caffee

class OpenCLSupportException: public std::exception {

public:
	OpenCLSupportException(std::string message) {
		message_ = message;
	}
	virtual ~OpenCLSupportException() throw() {
	}

	virtual const char* what() const throw() {
		return message_.c_str();
	}

protected:

private:
	std::string message_;
};


#endif // __OPENCL_SUPPORT_HPP__
#endif // USE_OPENCL
