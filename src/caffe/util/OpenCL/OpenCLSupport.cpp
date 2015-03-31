#ifdef USE_OPENCL

#include <glog/logging.h>
#include <CL/cl.h>
#include <clBLAS.h>
#include <caffe/util/OpenCL/OpenCLSupport.hpp>
#include <caffe/util/benchmark.hpp>
#include <caffe/syncedmem.hpp>
#include <typeinfo>
#include "limits.h"
#include <exception>

namespace caffe {

namespace OpenCL {

bool inititialized = false;
caffe::OpenCLManager mgr = OpenCLManager();
caffe::OpenCLPlatform* pf;
caffe::OpenCLDevice* gpu;
cl_context*	context;
cl_command_queue* queue;
cl_kernel* kernel;

const char* what(cl_int value) {
	int errorCode = (int) value;
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
		return "clBLAS An input dimension (M,N,K) is invalid";
	case clblasInvalidLeadDimA:
		return "clBLAS Leading dimension A must not be less than the size of the first dimension";
	case clblasInvalidLeadDimB:
		return "clBLAS Leading dimension B must not be less than the size of the second dimension";
	case clblasInvalidLeadDimC:
		return "clBLAS Leading dimension C must not be less than the size of the third dimension";
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


		//case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR:
		//    return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		//case CL_PLATFORM_NOT_FOUND_KHR:
		//    return "CL_PLATFORM_NOT_FOUND_KHR";
		//case CL_INVALID_PROPERTY_EXT:
		//    return "CL_INVALID_PROPERTY_EXT";
		//case CL_DEVICE_PARTITION_FAILED_EXT:
		//    return "CL_DEVICE_PARTITION_FAILED_EXT";
		//case CL_INVALID_PARTITION_COUNT_EXT:
		//    return "CL_INVALID_PARTITION_COUNT_EXT";
	default:
		return "unknown error code";
	}

	return "unknown error code";
}

int64_t getTickCount() {
	timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_nsec;
}

double getTickFrequency() {
	timespec t;
	clock_getres(CLOCK_MONOTONIC, &t);
	return (double) t.tv_nsec;
}

bool init() {

	if ( caffe::OpenCL::inititialized ) {
		LOG(INFO)<<"OpenCL already initialized";
		return true;
	}

	LOG(INFO)<<"Initialize OpenCL";
	mgr.query();

	if ( ! mgr.query() ) {
		LOG(ERROR) << "Failed to query OpenCL runtime information";
		return false;
	}

	if ( mgr.getNumPlatforms() <= 0 ) {
		LOG(ERROR) << "No OpenCL platforms found.";
		return false;
	}

	pf = mgr.getPlatform(0);
	if ( ! pf ) {
		LOG(ERROR) << "Failed to get first platform.";
		return false;
	}

	if ( ! pf->createContext() ) {
		LOG(ERROR) << "failed to create OpenCL context for platform " << pf->name();
		return false;
	}

	std::vector<std::string> cl_files;
	cl_files.push_back("src/caffe/util/OpenCL/math_functions.cl");
	cl_files.push_back("src/caffe/util/OpenCL/im2col.cl");
	cl_files.push_back("src/caffe/layers/OpenCL/pooling_layer.cl");
	cl_files.push_back("src/caffe/layers/OpenCL/relu_layer.cl");
	cl_files.push_back("src/caffe/layers/OpenCL/prelu_layer.cl");
	cl_files.push_back("src/caffe/layers/OpenCL/sigmoid_layer.cl");
	cl_files.push_back("src/caffe/layers/OpenCL/tanh_layer.cl");
	cl_files.push_back("src/caffe/layers/OpenCL/dropout_layer.cl");
	cl_files.push_back("src/caffe/layers/OpenCL/bnll_layer.cl");
	cl_files.push_back("src/caffe/layers/OpenCL/contrastive_loss_layer.cl");
	cl_files.push_back("src/caffe/layers/OpenCL/eltwise_layer.cl");
	cl_files.push_back("src/caffe/layers/OpenCL/lrn_layer.cl");
	cl_files.push_back("src/caffe/layers/OpenCL/softmax_layer.cl");
	cl_files.push_back("src/caffe/layers/OpenCL/softmax_loss_layer.cl");
	cl_files.push_back("src/caffe/layers/OpenCL/threshold_layer.cl");
	cl_files.push_back("src/caffe/layers/OpenCL/mvn_layer.cl");

	std::vector<std::string>::iterator it;

	for ( it = cl_files.begin(); it != cl_files.end(); it++ ) {
		if ( ! pf->compile(*it) ) {
			LOG(ERROR) << "failed to create to create OpenCL program for platform " << pf->name();
			return false;
		}
	}

	if ( pf->getNumGPUDevices() < 1 ) {
		LOG(ERROR) << "No GPU devices available at platform " << pf->name();
		return false;
	}

	gpu = pf->getDevice(CL_DEVICE_TYPE_GPU, 0);
	if ( ! gpu ) {
		LOG(ERROR) << "failed to select first GPU on platform " << pf->name();
		return false;
	}

	if ( ! gpu->createQueue() ) {
		LOG(ERROR) << "failed to create OpenCL command queue for device " << gpu->name();
		return false;
	}


	if ( clblasSetup() != CL_SUCCESS ) {
		LOG(ERROR) << "failed to initialize clBlas";
		return false;
	}

	gpu->print();
	caffe::OpenCL::inititialized = true;

	return true;
}

template<typename T>
std::string clGetKernelName(std::string name) {

	std::stringstream ss;
	ss<<name;

	if ( typeid(T) == typeid(float) ) {
		ss<<"Float";
	}
	if ( typeid(T) == typeid(double) ) {
		ss<<"Double";
	}
	if ( typeid(T) == typeid(char) ) {
		ss<<"Char";
	}
	if ( typeid(T) == typeid(int) ) {
		ss<<"Int";
	}

	return ss.str();
}
template std::string clGetKernelName<double>(std::string name);
template std::string clGetKernelName<float>(std::string name);

bool clMalloc(void** virtualPtr, size_t size) {

	OpenCLMemory clMem;
	try {
		clMem = OpenCLMemory(size);
	} catch (std::exception& e) {
		LOG(ERROR)<<e.what();
		return false;
	}
	gpu->add(clMem);

	*virtualPtr = clMem.getVirtualPointer();
	return true;
}

bool clFree(void* virtualPtr) {

	if ( ! gpu->isValidPtr(virtualPtr) ) {
		LOG(ERROR) << gpu->name() << "> not a valid memory pointer @ " << virtualPtr;
		return false;
	}

	OpenCLMemory clMem;
	if ( ! gpu->get(virtualPtr, clMem) ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCLMemory object using virtual pointer @ " << virtualPtr;
		return false;
	}

	/*

	cl_mem* logicalPtr = clMem.getLogicalPointer2();
	DLOG(INFO)<<"cl_mem* = "<<logicalPtr;
	DLOG(INFO)<<"cl_mem  = "<<*logicalPtr;

	if ( logicalPtr == NULL ) {
		LOG(ERROR) << gpu->name() << "> received logical NULL pointer from virtual pointer @ " << virtualPtr;
		return false;
	}

	if ( ! CL_CHECK(clReleaseMemObject(*logicalPtr) ) ) {
		return false;
	}

	DLOG(INFO) << gpu->name() << "> release OpenCL memory object at "<<gpu->getMemoryTag(virtualPtr).c_str();
	*/
	gpu->rmMemoryPtr(clMem.getVirtualPointer());

	queue = gpu->getQueue();
	if ( queue != NULL ) {
		//clFinish(*queue);
	}
	return true;
}

template<typename T>
bool clMemset(void* virtualPtr, const T alpha, const size_t Bytes) {

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

#ifdef OPENCL_VERSION_1_2 // at least OpenCL Version 1.2 required

	if ( ! gpu->isValidPtr(virtualPtr) ) {
		LOG(ERROR) << gpu->name() << "> not a valid memory pointer @ " << virtualPtr;
		return false;
	}

	OpenCLMemory clMem;
	if ( ! gpu->get(virtualPtr, clMem) ) {
		LOG(ERROR) << gpu->name() << "> failed to get GPU memory @ " << virtualPtr;
		return false;
	}
	size_t mem_offset	= clGetMemoryOffset(virtualPtr);
	cl_mem base 			= (cl_mem) clMem.getLogicalPointer();

	if ( ! CL_CHECK(clEnqueueFillBuffer(*queue, base, &alpha, sizeof(T), mem_offset, Bytes, 0, NULL, NULL) ) ) {
		return false;
	}
#else
	std::string kernel_name = clGetKernelName<T>("clFillBuffer");

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	unsigned int N = Bytes/sizeof(T);
	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, N)
	CL_SET_TYPE_KERNEL_ARG(T, alpha)
	CL_SET_ARRAY_KERNEL_ARG(&virtualPtr)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(N, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(N, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		std::ostringstream oss;
		oss << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<what(err);
		throw OpenCLSupportException(oss.str());
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END
#endif


	DLOG(INFO) << gpu->name() << "> set OpenCL memory at "<<gpu->getMemoryTag(virtualPtr).c_str()<<" to "<<alpha;
	return true;
}
template bool clMemset<char>(void* gpuPtr, const char alpha, const size_t N);
template bool clMemset<int>(void* gpuPtr, const int alpha, const size_t N);
template bool clMemset<float>(void* gpuPtr, const float alpha, const size_t N);
template bool clMemset<double>(void* gpuPtr, const double alpha, const size_t N);

template<typename T>
bool clGPU2GPU(const void* src, void* dst, size_t Bytes) {

	if ( ! gpu->isValidPtr(src) ) {
		LOG(ERROR) << gpu->name() << "> not a valid memory pointer @ " << src;
		return false;
	}

	OpenCLMemory clMemSrc;
	if ( ! gpu->get(src, clMemSrc) ) {
		LOG(ERROR) << gpu->name() << "> failed to get GPU memory @ " << src;
		return false;
	}
	size_t offsetSrc	= clGetMemoryOffset(src);

	if ( ! gpu->isValidPtr(dst) ) {
		LOG(ERROR) << gpu->name() << "> not a valid memory pointer @ " << dst;
		return false;
	}

	OpenCLMemory clMemDst;
	if ( ! gpu->get(dst, clMemDst) ) {
		LOG(ERROR) << gpu->name() << "> failed to get GPU memory @ " << dst;
		return false;
	}
	size_t offsetDst	= clGetMemoryOffset(dst);

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	std::string kernel_name = clGetKernelName<T>("clGPU2GPU");

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	unsigned int N = Bytes/sizeof(T);
	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, N)
	CL_SET_ARRAY_KERNEL_ARG(&src)
	CL_SET_TYPE_KERNEL_ARG(int, (int) offsetSrc)
	CL_SET_ARRAY_KERNEL_ARG(&dst)
	CL_SET_TYPE_KERNEL_ARG(int, (int) offsetDst)

	size_t global = 1024;//CAFFE_GET_GLOBAL_WORKITEMS(N, OPENCL_LOCAL_SIZE);
	size_t local  = 1024;//CAFFE_GET_LOCAL_WORKITEMS(N, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		std::ostringstream oss;
		oss << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<what(err);
		throw OpenCLSupportException(oss.str());
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END


	return true;
}
template bool clGPU2GPU<char>(const void* src, void* dst, size_t Bytes);
template bool clGPU2GPU<int>(const void* src, void* dst, size_t Bytes);
template bool clGPU2GPU<float>(const void* src, void* dst, size_t Bytes);
template bool clGPU2GPU<double>(const void* src, void* dst, size_t Bytes);


bool clMemcpy(void* virtualDstPtr, const void* virtualSrcPtr, size_t size, int type) {

	std::string function = __func__;

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	OpenCLMemory clMemSrc, clMemDst, mem1, mem2, tmp;
	const void *baseSrc;
	const void* baseDst;
	size_t offsetSrc, offsetDst;

	switch(type) {

	case caffe::OpenCL::COPY_CPU_TO_CPU:
		/*
		if ( gpu->isValidPtr(src) ) {
			LOG(ERROR) << gpu->name() << "> src pointer is in GPU memory @ " << src;
			return false;
		}
		*/
		if ( gpu->isValidPtr(virtualDstPtr) ) {
			LOG(ERROR) << gpu->name() << "> dst pointer is in GPU memory @ " << virtualDstPtr;
			return false;
		}
		memcpy(virtualDstPtr, virtualSrcPtr, size);
		DLOG(INFO) << gpu->name() << "> copy CPU@"<<virtualSrcPtr<<" to CPU@"<<virtualDstPtr<<" "<<size<<" Byte transferred.";
		break;

	case caffe::OpenCL::COPY_CPU_TO_GPU:
		/*
		if ( gpu->isValidPtr(src) ) {
			LOG(ERROR) << gpu->name() << "> src pointer is in GPU memory @ " << src;
			return false;
		}
		*/
		if ( ! gpu->isValidPtr(virtualDstPtr) ) {
			LOG(ERROR) << gpu->name() << "> dst pointer is not in GPU memory @ " << virtualDstPtr;
			return false;
		}

		if ( ! gpu->get(virtualDstPtr, clMemDst) ) {
			LOG(ERROR) << gpu->name() << "> failed to get GPU memory @ " << virtualDstPtr;
			return false;
		}
		baseDst 	= clMemDst.getLogicalPointer();
		offsetDst	= clGetMemoryOffset(virtualDstPtr);

		if ( ! CL_CHECK(clEnqueueWriteBuffer(*queue, (cl_mem) baseDst, CL_TRUE, offsetDst, size, virtualSrcPtr, 0, NULL, NULL) ) ) {
			LOG(ERROR) << gpu->name() << "> copy CPU@"<<virtualSrcPtr<<" to "<<gpu->getMemoryTag(virtualDstPtr).c_str()<<" "<<size<<" Byte transferred failed.";
			return false;
		};
		//clFinish(*queue);
		DLOG(INFO) << gpu->name() << "> copy CPU@"<<virtualSrcPtr<<" to "<<gpu->getMemoryTag(virtualDstPtr).c_str()<<" "<<size<<" Byte transferred.";
		break;

	case caffe::OpenCL::COPY_GPU_TO_CPU:
		if ( ! gpu->isValidPtr(virtualSrcPtr) ) {
			LOG(ERROR) << gpu->name() << "> src pointer is not in GPU memory @ " << virtualSrcPtr;
			return false;
		}
		/*
		if ( gpu->isValidPtr(dst) ) {
			LOG(ERROR) << gpu->name() << "> dst pointer is in GPU memory @ " << dst;
			return false;
		}
		*/
		if ( ! gpu->get(virtualSrcPtr, clMemSrc) ) {
			LOG(ERROR) << gpu->name() << "> failed to get GPU memory @ " << virtualSrcPtr;
			return false;
		}
		baseSrc 	= clMemSrc.getLogicalPointer();
		offsetSrc	= clGetMemoryOffset(virtualSrcPtr);

		if ( ! CL_CHECK(clEnqueueReadBuffer(*queue, (cl_mem) baseSrc, CL_TRUE, offsetSrc, size, virtualDstPtr, 0, NULL, NULL) ) ) {
			return false;
		};
		//clFinish(*queue);
		DLOG(INFO) << gpu->name() << "> copy "<<gpu->getMemoryTag(virtualSrcPtr).c_str()<<" to CPU@"<<virtualDstPtr<<" "<<size<<" Byte transferred.";
		break;

	case caffe::OpenCL::COPY_GPU_TO_GPU:
		if ( ! gpu->isValidPtr(virtualSrcPtr) ) {
			LOG(ERROR) << gpu->name() << "> src pointer is not in GPU memory @ " << virtualSrcPtr;
			return false;
		}
		if ( ! gpu->isValidPtr(virtualDstPtr) ) {
			LOG(ERROR) << gpu->name() << "> dst pointer is not in GPU memory @ " << virtualDstPtr;
			return false;
		}
		DLOG(INFO) << "VMS@"<<virtualSrcPtr<<" to VMD@"<<virtualDstPtr<<" size = "<<size;

		if ( ! gpu->get(virtualSrcPtr, clMemSrc) ) {
			LOG(ERROR) << gpu->name() << "> failed to get GPU memory @ " << virtualSrcPtr;
			return false;
		}
		baseSrc 	= clMemSrc.getLogicalPointer();
		offsetSrc	= clGetMemoryOffset(virtualSrcPtr);
		DLOG(INFO) << gpu->getMemoryTag(clMemSrc.getVirtualPointer()).c_str();
		DLOG(INFO) << "VMS@"<<baseSrc<<" offset = "<<offsetSrc;

		if ( ! gpu->get(virtualDstPtr, clMemDst) ) {
			LOG(ERROR) << gpu->name() << "> failed to get GPU memory @ " << virtualDstPtr;
			return false;
		}
		baseDst 	= clMemDst.getLogicalPointer();
		offsetDst	= clGetMemoryOffset(virtualDstPtr);
		DLOG(INFO) << gpu->getMemoryTag(clMemDst.getVirtualPointer()).c_str();
		DLOG(INFO) << "VMD@"<<baseDst<<" offset = "<<offsetDst;

		if ( offsetSrc + size > clMemSrc.getSize() ) {
			LOG(ERROR) << gpu->name() << "> copy range out of source range.";
			return false;
		}

		if ( offsetDst + size > clMemDst.getSize() ) {
			LOG(ERROR) << gpu->name() << "> copy range out of destination range.";
			return false;
		}

		/*
		DLOG(INFO)<<"baseSrc@"<<baseSrc<<" baseDst@"<<baseDst;
		DLOG(INFO)<<(baseSrc) + (offsetSrc) + (size);
		DLOG(INFO)<<(baseDst) + (offsetDst);
		DLOG(INFO)<<(baseSrc+offsetSrc+size < baseDst+offsetDst);
		DLOG(INFO)<<(baseDst) + (offsetDst) + (size);
		DLOG(INFO)<<(baseSrc) + (offsetSrc);
		DLOG(INFO)<<(baseDst+offsetDst+size > baseSrc+offsetSrc);
		*/
		//if ( ! ((baseSrc+offsetSrc+size < baseDst+offsetDst) || (baseDst+offsetDst+size > baseSrc+offsetSrc)) ) {
		if ( clMemSrc.getVirtualPointer() != clMemDst.getVirtualPointer() ) {

			DLOG(INFO) << "no overlap";
			/*
			DLOG(INFO) << gpu->getMemoryTag(baseSrc).c_str();
			DLOG(INFO) << gpu->getMemoryTag(baseDst).c_str();
			DLOG(INFO) << "offset Src = "<<offsetSrc;
			DLOG(INFO) << "offset Dst = "<<offsetDst;
			DLOG(INFO) << "size       = "<<size;
			*/

			if ( ! CL_CHECK( clEnqueueCopyBuffer(*queue, (cl_mem) baseSrc, (cl_mem) baseDst, offsetSrc, offsetDst, size, 0, NULL, NULL) ) ) {
				return false;
			}
			//clFinish(*queue);
		} else {
			DLOG(INFO) << "caffe::OpenCL::COPY_GPU_TO_GPU: with overlap";

			//clGPU2GPU<char>(clMemSrc.getVirtualPointer(), clMemDst.getVirtualPointer(), size);

			void* bufVPtr = NULL;
			if ( ! clMalloc(&bufVPtr, size) ) {
				LOG(ERROR) << gpu->name() << "> failed to created buffer of size = " << size << " Byte";
				return false;
			}

			OpenCLMemory buffer;
			if ( ! gpu->get(bufVPtr, buffer) ) {
				LOG(ERROR) << gpu->name() << "> failed to get buffer Object GPU@VIRT"<<bufVPtr;
				return false;
			}
			const void* bufLPtr = buffer.getLogicalPointer();

			/*
			DLOG(INFO) << "src = "<<gpu->getMemoryTag(virtualSrcPtr).c_str();
			DLOG(INFO) << "buf = "<<gpu->getMemoryTag(bufVPtr).c_str();
			DLOG(INFO) << "dst = "<<gpu->getMemoryTag(virtualDstPtr).c_str();
			DLOG(INFO) << "offset Src = "<<offsetSrc;
			DLOG(INFO) << "offset Dst = "<<offsetDst;
			DLOG(INFO) << "size       = "<<size;
			*/

			if ( ! CL_CHECK( clEnqueueCopyBuffer(*queue, (cl_mem) baseSrc, (cl_mem) bufLPtr, offsetSrc, 0, size, 0, NULL, NULL) ) ) {
				return false;
			}
			DLOG(INFO) << gpu->name() << "> copy "<<gpu->getMemoryTag(virtualSrcPtr).c_str()<<" with offset " << offsetSrc << " to buffer " <<gpu->getMemoryTag(bufVPtr).c_str()<< " "<<size<<" Byte transferred.";

			if ( ! CL_CHECK( clEnqueueCopyBuffer(*queue, (cl_mem) bufLPtr, (cl_mem) baseDst, 0, offsetDst, size, 0, NULL, NULL) ) ) {
				return false;
			}
			DLOG(INFO) << gpu->name() << "> copy "<<gpu->getMemoryTag(bufVPtr).c_str()<<" to "<<gpu->getMemoryTag(virtualDstPtr).c_str()<<" with offset "<<offsetDst<<" "<<size<<" Byte transferred.";

			clFree(bufVPtr);
		}

		DLOG(INFO) << gpu->name() << "> copy "<<gpu->getMemoryTag(virtualSrcPtr).c_str()<<" with offset " << offsetSrc << " to "<<gpu->getMemoryTag(virtualDstPtr).c_str()<<" with offset " << offsetDst << " "<<size<<" Byte transferred.";
		break;

	case caffe::OpenCL::COPY_DEFAULT:
		if ( ! gpu->isValidPtr(virtualSrcPtr) && ! gpu->isValidPtr(virtualDstPtr) ) {
			return clMemcpy(virtualDstPtr, virtualSrcPtr, size, caffe::OpenCL::COPY_CPU_TO_CPU);
		}
		if ( ! gpu->isValidPtr(virtualSrcPtr) && gpu->isValidPtr(virtualDstPtr) ) {
			return clMemcpy(virtualDstPtr, virtualSrcPtr, size, caffe::OpenCL::COPY_CPU_TO_GPU);
		}
		if ( gpu->isValidPtr(virtualSrcPtr) && ! gpu->isValidPtr(virtualDstPtr) ) {
			return clMemcpy(virtualDstPtr, virtualSrcPtr, size, caffe::OpenCL::COPY_GPU_TO_CPU);
		}
		if ( gpu->isValidPtr(virtualSrcPtr) && gpu->isValidPtr(virtualDstPtr) ) {
			return clMemcpy(virtualDstPtr, virtualSrcPtr, size, caffe::OpenCL::COPY_GPU_TO_GPU);
		}
		break;

	default:
		LOG(ERROR) << "unsupported copy mode = " << type;
		return false;
		break;
	}

	return true;
}

bool clReleaseSubBuffers(std::vector<cl_mem>& subBuffers) {

	for( std::vector<cl_mem>::iterator it = subBuffers.begin(); it != subBuffers.end(); it++ ) {
		cl_int err = clReleaseMemObject(*it);
		if ( err != CL_SUCCESS ) {
			LOG(ERROR)<<"failed to release sub-buffer";
			return false;
		}
	}
	return true;
}

bool clReleaseBufferMap(std::map<const void*, std::pair<void*, size_t> >& bufferMap) {

	for( std::map<const void*, std::pair<void*, size_t> >::iterator it = bufferMap.begin(); it != bufferMap.end(); it++ ) {
		const void* virtual_ptr = it->first;
		void* buffer_ptr  		= bufferMap[virtual_ptr].first;
		size_t buffer_size 		= bufferMap[virtual_ptr].second;
		if ( ! clMemcpy(const_cast<void*>(virtual_ptr), buffer_ptr, buffer_size, caffe::OpenCL::COPY_DEFAULT) ) {
			return false;
		}
		clFree((void*) buffer_ptr);
	}
	return true;
}

bool clSetKernelArrayArg(const void* ptr_virtual, unsigned int& idx, std::vector<cl_mem>& subBuffers, std::map<const void*, std::pair<void*, size_t> >& bufferMap) {

	cl_int err;

	/* case when array is NULL */
	if  ( ptr_virtual == NULL ) {
		DLOG(INFO)<<"kernel variable pointer is NULL for kernel argument "<<idx;
		err = clSetKernelArg(*kernel, idx, sizeof(cl_mem), NULL);
		if ( err != CL_SUCCESS ) {
			LOG(ERROR) << "failed to set kernel argument "<<idx<<" for kernel on GPU "<<gpu->name()<<" : "<<what(err);
			return false;
		}
		idx++;
		return true;
	}

	/* get cl_mem location using the virtual memory pointer */
	const void* ptr_logical;
	if ( ! clMakeLogical(ptr_virtual, &ptr_logical) ) {
		return false;
	}

	size_t offset	= clGetMemoryOffset(ptr_virtual);
	size_t size 	= clGetMemorySize(ptr_virtual);

	/* case when there is no offset to base of cl_mem */
	if ( offset == 0 ) {
		err = clSetKernelArg(*kernel, idx, sizeof(ptr_logical), &ptr_logical);
		if ( err != CL_SUCCESS ) {
			LOG(ERROR) << "failed to set kernel argument "<<idx<<" for kernel on GPU "<<gpu->name()<<" : "<<what(err);
			return false;
		}
		idx++;
		return true;
	}

	/* check alignment */
	cl_uint bytes = gpu->getDeviceMemBaseAddrAlign();
	if ( offset % bytes != 0 ) {
		LOG(WARNING)<<"sub-buffer memory offset ("<<offset<<" Byte) is not aligned with device memory offset ("<<bytes<<" Byte)";

		cl_mem_flags flags;
		err = clGetMemObjectInfo((cl_mem) ptr_logical, CL_MEM_FLAGS, sizeof(flags), &flags, NULL);
		if ( err != CL_SUCCESS ) {
			LOG(ERROR)<<"failed to query memory properties.";
			return false;
		}

		void* ptr_buffer;
		size_t buffer_size = size-offset;
		if ( ! clMalloc(&ptr_buffer, buffer_size) ) {
			return false;
		}

		if ( ! clMemcpy(ptr_buffer, ptr_virtual, buffer_size, caffe::OpenCL::COPY_DEFAULT) ) {
			return false;
		}

		if ( ! clMakeLogical(ptr_buffer, &ptr_logical) ) {
			return false;
		}
		bufferMap[ptr_virtual] = std::make_pair(ptr_buffer, buffer_size);

		err = clSetKernelArg(*kernel, idx, sizeof(ptr_logical), &ptr_logical);
		if ( err != CL_SUCCESS ) {
			LOG(ERROR) << "failed to set kernel argument "<<idx<<" for kernel on GPU "<<gpu->name()<<" : "<<what(err);
			return false;
		}
		idx++;
		return true;
	}

	/* case when there is an offset to base of cl_mem */
	DLOG(INFO)<<"memory offset = "<<offset<<" detected, create sub-buffer ["<<offset<<"|"<<size-offset<<"]";
	cl_buffer_region region = {offset, size - offset};
	cl_mem sb = clCreateSubBuffer((cl_mem) ptr_logical, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "failed to create sub-buffer for kernel argument "<<idx<<" for kernel on GPU "<<gpu->name()<<" : "<<what(err);
		return false;
	}
	DLOG(INFO)<<"create sub-buffer "<<subBuffers.size()<<std::endl;
	subBuffers.push_back(sb);

	err = clSetKernelArg(*kernel, idx, sizeof(sb), &sb);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "failed to set kernel argument "<<idx<<" for kernel on GPU "<<gpu->name()<<" : "<<what(err);
		return false;
	}
	idx++;

	return true;
}

template<typename T>
bool clSetKernelTypeArg(T variable, unsigned int& idx) {

	cl_int err;

	err = clSetKernelArg(*kernel, idx, sizeof(T), &variable);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "failed to set kernel argument "<<idx<<" for kernel on GPU "<<gpu->name()<<" : "<<what(err);\
		return false;
	}
	idx++;
	return true;
}
template bool clSetKernelTypeArg<int>(int variable, unsigned int& idx);
template bool clSetKernelTypeArg<const int>(const int variable, unsigned int& idx);
template bool clSetKernelTypeArg<unsigned int>(unsigned int variable, unsigned int& idx);
template bool clSetKernelTypeArg<const unsigned int>(const unsigned int variable, unsigned int& idx);
template bool clSetKernelTypeArg<float>(float variable, unsigned int& idx);
template bool clSetKernelTypeArg<const float>(const float variable, unsigned int& idx);
template bool clSetKernelTypeArg<double>(double variable, unsigned int& idx);
template bool clSetKernelTypeArg<const double>(const double variable, unsigned int& idx);

template<typename T>
bool clBLASasum(const int N, const void* array_virtual, T* y) {

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	caffe::SyncedMemory asum(sizeof(T));
	const void* asum_virtual = asum.mutable_gpu_data();
	const void* asum_logical;
	if ( ! clMakeLogical(asum_virtual, &asum_logical) ) {
		return false;
	}

	caffe::SyncedMemory buf(N*sizeof(T));
	const void* buf_virtual	= buf.mutable_gpu_data();
	const void* buf_logical;
	if ( ! clMakeLogical(buf_virtual, &buf_logical) ) {
		return false;
	}

	const void* array_logical;
	if ( ! clMakeLogical(array_virtual, &array_logical) ) {
		return false;
	}

	if ( typeid(T) == typeid(float) ) {
		if ( ! CL_CHECK(clblasSasum(N, (cl_mem) asum_logical, 0, (cl_mem) array_logical, 0, 1, (cl_mem) buf_logical, 1, queue, 0, NULL, NULL)) ) {
			return false;
		}
	}

	if ( typeid(T) == typeid(double) ) {
		if ( ! CL_CHECK(clblasDasum(N, (cl_mem) asum_logical, 0, (cl_mem) array_logical, 0, 1, (cl_mem) buf_logical, 1, queue, 0, NULL, NULL)) ) {
			return false;
		}
	}

	*y = *((T*) asum.mutable_cpu_data());
	return true;
}
template bool clBLASasum<float>(const int N, const void* array_GPU_ptr, float* y);
template bool clBLASasum<double>(const int N, const void* array_GPU_ptr, double* y);

template<typename T>
bool clsign(const int n, const void* array_GPU_x, void* array_GPU_y) {

	std::string kernel_name = clGetKernelName<T>("clsign");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, n)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_y)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clsign<float>(const int n, const void* array_GPU_x, void* array_GPU_y);
template bool clsign<double>(const int n, const void* array_GPU_x, void* array_GPU_y);

template<typename T>
bool clsgnbit(const int n, const void* array_GPU_x, void* array_GPU_y) {

	std::string kernel_name = clGetKernelName<T>("clsgnbit");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, n)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_y)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clsgnbit<float>(const int n, const void* array_GPU_x, void* array_GPU_y);
template bool clsgnbit<double>(const int n, const void* array_GPU_x, void* array_GPU_y);

template<typename T>
bool clabs(const int n, const void* array_GPU_x, void* array_GPU_y) {

	std::string kernel_name = clGetKernelName<T>("clabs");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, n)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_y)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clabs<float>(const int n, const void* array_GPU_x, void* array_GPU_y);
template bool clabs<double>(const int n, const void* array_GPU_x, void* array_GPU_y);

template<typename T>
bool cldiv(const int n, const void* array_GPU_x, const void* array_GPU_y, void* array_GPU_z) {

	std::string kernel_name = clGetKernelName<T>("cldiv");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, n)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_y)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_z)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool cldiv<float>(const int n, const void* array_GPU_x, const void* array_GPU_y, void* array_GPU_z);
template bool cldiv<double>(const int n, const void* array_GPU_x, const void* array_GPU_y, void* array_GPU_z);

template<typename T>
bool clmul(const int n, const void* array_GPU_x, const void* array_GPU_y, void* array_GPU_z) {

	std::string kernel_name = clGetKernelName<T>("clmul");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, n)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_y)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_z)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clmul<float>(const int n, const void* array_GPU_x, const void* array_GPU_y, void* array_GPU_z);
template bool clmul<double>(const int n, const void* array_GPU_x, const void* array_GPU_y, void* array_GPU_z);

template<typename T>
bool clBLASscal(const int n, const float alpha, const void* array_x_virtual, void* array_y_virtual) {

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	const void* array_x_logical;
	if ( ! clMakeLogical(array_x_virtual, &array_x_logical) ) {
		return false;
	}

	const void* array_y_logical;
	if ( ! clMakeLogical(array_y_virtual, &array_y_logical) ) {
		return false;
	}


	if( typeid(T) == typeid(float) ) {
		if ( ! CL_CHECK( clblasScopy(n, (cl_mem) array_x_logical, 0, 1, (cl_mem) array_y_logical, 0, 1, 1, queue, 0, NULL, NULL) ) ) {
			LOG(ERROR) << "clblasScopy() failed on GPU "<<gpu->name();
			return false;
		}
		if ( ! CL_CHECK( clblasSscal(n, alpha, (cl_mem) array_y_logical, 0, 1, 1, queue, 0, NULL, NULL) ) ) {
			LOG(ERROR) << "clblasSscal() failed on GPU "<<gpu->name();
			return false;
		}
	}

	if( typeid(T) == typeid(double) ) {
		if ( ! CL_CHECK( clblasDcopy(n, (cl_mem) array_x_logical, 0, 1, (cl_mem) array_y_logical, 0, 1, 1, queue, 0, NULL, NULL) ) ) {
			LOG(ERROR) << "clblasDcopy() failed on GPU "<<gpu->name();
			return false;
		}
		if ( ! CL_CHECK( clblasDscal(n, alpha, (cl_mem) array_y_logical, 0, 1, 1, queue, 0, NULL, NULL) ) ) {
			LOG(ERROR) << "clblasDscal() failed on GPU "<<gpu->name();
			return false;
		}
	}

	return true;
}
template bool clBLASscal<float>(const int n, const float alpha, const void* array_x_virtual, void* array_y_virtual);
template bool clBLASscal<double>(const int n, const float alpha, const void* array_x_virtual, void* array_y_virtual);

template<typename T>
bool clBLASdot(const int n, const T* x, const int incx, const T* y, const int incy, T* out) {

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	/*
	OpenCLMemory clMem;
	if ( ! gpu->get(x, clMem) ) {
		return false;
	}
	DLOG(INFO)<<"x : "<<clMem.getTag().c_str();
	*/
	std::vector<cl_mem> sb;
	std::map<const void*, std::pair<void*, size_t> > bm;

	const void* x_logical;
	if ( ! clMakeLogical2(x, &x_logical, sb, bm) ) {
		return false;
	}

	/*
	if ( ! gpu->get(y, clMem) ) {
		return false;
	}
	DLOG(INFO)<<"y : "<<clMem.getTag().c_str();
	*/
	const void* y_logical;
	if ( ! clMakeLogical2(y, &y_logical, sb, bm) ) {
		return false;
	}

	caffe::SyncedMemory dot(sizeof(T));
	const void* dot_virtual = dot.mutable_gpu_data();
	const void* dot_logical;
	if ( ! clMakeLogical2(dot_virtual, &dot_logical, sb, bm) ) {
		return false;
	}

	caffe::SyncedMemory buf(n*sizeof(T));
	const void* buf_virtual = buf.mutable_gpu_data();
	const void* buf_logical;
	if ( ! clMakeLogical2(buf_virtual, &buf_logical, sb, bm) ) {
		return false;
	}

	if( typeid(T) == typeid(float) ) {
		if ( ! CL_CHECK( clblasSdot(n, (cl_mem) dot_logical, 0, (cl_mem) x_logical, 0, 1, (cl_mem) y_logical, 0, 1, (cl_mem) buf_logical, 1, queue, 0, NULL, NULL) ) ) {
			LOG(ERROR) << "clblasSdot() failed on GPU "<<gpu->name();
			clReleaseSubBuffers(sb);
			clReleaseBufferMap(bm);
			return false;
		}
		DLOG(INFO) << "clblasSdot() succeeded on GPU "<<gpu->name();
	}

	if( typeid(T) == typeid(double) ) {
		if ( ! CL_CHECK( clblasDdot(n, (cl_mem) dot_logical, 0, (cl_mem) x_logical, 0, 1, (cl_mem) y_logical, 0, 1, (cl_mem) buf_logical, 1, queue, 0, NULL, NULL) ) ) {
			LOG(ERROR) << "clblasDdot() failed on GPU "<<gpu->name();
			clReleaseSubBuffers(sb);
			clReleaseBufferMap(bm);
			return false;
		}
		DLOG(INFO) << "clblasDdot() succeeded on GPU "<<gpu->name();
	}

	clReleaseSubBuffers(sb);
	clReleaseBufferMap(bm);
	*out = *((T*) dot.mutable_cpu_data());
	return true;
}
template bool clBLASdot<float>(const int n, const float* x, const int incx, const float* y, const int incy, float* out);
template bool clBLASdot<double>(const int n, const double* x, const int incx, const double* y, const int incy, double* out);

template<typename T>
bool clBLASgemv(const clblasTranspose TransA, const int m, const int n, const T alpha, const T* A, const T* x, const T beta, T* y) {

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

  void*   A_base    = clGetMemoryBase(A);
  if ( ! A_base ) {
    LOG(ERROR)<<"failed to get OpenCL device memory address for virtual address @ "<<A;
    return false;
  }
  const void* A_device = NULL;
  if ( ! clMakeLogical(A_base, &A_device) ) {
    return false;
  }
  size_t  A_offset  = clGetMemoryOffset(A);
  if ( A_offset < 0 ) {
    LOG(ERROR)<<"failed to get OpenCL device memory offset for virtual address @ "<<A;
    return false;
  }
  A_offset /= sizeof(T);
  //DLOG(INFO)<<A<<" -> "<<A_base<<" with offset = "<<A_offset;

  void*   x_base    = clGetMemoryBase(x);
  if ( ! x_base ) {
    LOG(ERROR)<<"failed to get OpenCL device memory address for virtual address @ "<<x;
    return false;
  }
  const void* x_device = NULL;
  if ( ! clMakeLogical(x_base, &x_device) ) {
    return false;
  }
  size_t  x_offset  = clGetMemoryOffset(x);
  if ( x_offset < 0 ) {
    LOG(ERROR)<<"failed to get OpenCL device memory offset for virtual address @ "<<x;
    return false;
  }
  x_offset /= sizeof(T);
  //DLOG(INFO)<<x<<" -> "<<x_base<<" with offset = "<<x_offset;

  void*   y_base    = clGetMemoryBase(y);
  if ( ! y_base ) {
    LOG(ERROR)<<"failed to get OpenCL device memory address for virtual address @ "<<y;
    return false;
  }
  const void* y_device = NULL;
  if ( ! clMakeLogical(y_base, &y_device) ) {
    return false;
  }
  size_t  y_offset  = clGetMemoryOffset(y);
  if ( y_offset < 0 ) {
    LOG(ERROR)<<"failed to get OpenCL device memory offset for virtual address @ "<<y;
    return false;
  }
  y_offset /= sizeof(T);
  //DLOG(INFO)<<y<<" -> "<<y_base<<" with offset = "<<y_offset;


	/*
	std::vector<cl_mem> sb;
	std::map<const void*, std::pair<void*, size_t> > bm;

	const void* A_logical;
	if ( ! clMakeLogical2(A, &A_logical, sb, bm) ) {
		return false;
	}

	const void* x_logical;
	if ( ! clMakeLogical2(x, &x_logical, sb, bm) ) {
		return false;
	}

	const void* y_logical;
	if ( ! clMakeLogical2(y, &y_logical, sb, bm) ) {
		return false;
	}
  */

	if( typeid(T) == typeid(float) ) {
		if ( ! CL_CHECK( clblasSgemv(clblasRowMajor, TransA, m, n, alpha, (cl_mem) A_device, A_offset, n, (cl_mem) x_device, x_offset, 1, beta, (cl_mem) y_device, y_offset, 1, 1, queue, 0, NULL, NULL) ) ) {
			LOG(ERROR) << "clblasSgemv() failed on GPU "<<gpu->name();
			//clReleaseSubBuffers(sb);
			//clReleaseBufferMap(bm);
			return false;
		}
		DLOG(INFO) << "clblasSgemv() succeeded on GPU "<<gpu->name();
	}

	if( typeid(T) == typeid(double) ) {
		if ( ! CL_CHECK( clblasDgemv(clblasRowMajor, TransA, m, n, alpha, (cl_mem) A_device, A_offset, n, (cl_mem) x_device, x_offset, 1, beta, (cl_mem) y_device, y_offset, 1, 1, queue, 0, NULL, NULL) ) ) {
			LOG(ERROR) << "clblasDgemv() failed on GPU "<<gpu->name();
			//clReleaseSubBuffers(sb);
			//clReleaseBufferMap(bm);
			return false;
		}
		DLOG(INFO) << "clblasDgemv() succeeded on GPU "<<gpu->name();
	}
	//clReleaseSubBuffers(sb);
	//clReleaseBufferMap(bm);
	return true;
}
template bool clBLASgemv<float>(const clblasTranspose TransA, const int m, const int n, const float alpha, const float* A, const float* x, const float beta, float* y);
template bool clBLASgemv<double>(const clblasTranspose TransA, const int m, const int n, const double alpha, const double* A, const double* x, const double beta, double* y);

template<typename T>
bool clBLASgemv(const clblasTranspose TransA, const int m, const int n, const T alpha, const T* A, const int step_A, const T* x, const int step_x, const T beta, T* y, const int step_y) {

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	const void* A_logical;
	if ( ! clMakeLogical(A, &A_logical) ) {
		return false;
	}

	const void* x_logical;
	if ( ! clMakeLogical(x, &x_logical) ) {
		return false;
	}

	const void* y_logical;
	if ( ! clMakeLogical(y, &y_logical) ) {
		return false;
	}

	if( typeid(T) == typeid(float) ) {
		if ( ! CL_CHECK( clblasSgemv(clblasRowMajor, TransA, m, n, alpha, (cl_mem) A_logical, step_A, n, (cl_mem) x_logical, step_x, 1, beta, (cl_mem) y_logical, step_y, 1, 1, queue, 0, NULL, NULL) ) ) {
			LOG(ERROR) << "clblasSgemv() failed on GPU "<<gpu->name();
			return false;
		}
		DLOG(INFO) << "clblasSgemv() succeeded on GPU "<<gpu->name();
	}

	if( typeid(T) == typeid(double) ) {
		if ( ! CL_CHECK( clblasDgemv(clblasRowMajor, TransA, m, n, alpha, (cl_mem) A_logical, step_A, n, (cl_mem) x_logical, step_x, 1, beta, (cl_mem) y_logical, step_y, 1, 1, queue, 0, NULL, NULL) ) ) {
			LOG(ERROR) << "clblasDgemv() failed on GPU "<<gpu->name();
			return false;
		}
		DLOG(INFO) << "clblasDgemv() succeeded on GPU "<<gpu->name();
	}

	return true;
}
template bool clBLASgemv<float>(const clblasTranspose TransA, const int m, const int n, const float alpha, const float* A, const int step_A, const float* x, const int step_x, const float beta, float* y, const int step_y);
template bool clBLASgemv<double>(const clblasTranspose TransA, const int m, const int n, const double alpha, const double* A, const int step_A, const double* x, const int step_x, const double beta, double* y, const int step_y);

template<typename T>
bool clgemm(const int m, const int n, const int k, const T* A, const T* B, T* C) {

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	std::string kernel_name = clGetKernelName<T>("mmul");

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, m)
	CL_SET_TYPE_KERNEL_ARG(int, n)
	CL_SET_TYPE_KERNEL_ARG(int, k)
	CL_SET_ARRAY_KERNEL_ARG(&A)
	CL_SET_ARRAY_KERNEL_ARG(&B)
	CL_SET_ARRAY_KERNEL_ARG(&C)
	clSetKernelArg(*kernel, 6, sizeof(T)*k, NULL);

	size_t global = n;//CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
	size_t local  = 1;//CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		std::ostringstream oss;
		oss << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<what(err);
		throw OpenCLSupportException(oss.str());
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clgemm<float>(const int m, const int n, const int k, const float* A, const float* B, float* C);
template bool clgemm<double>(const int m, const int n, const int k, const double* A, const double* B, double* C);

template<typename T>
bool clBLASgemm(const clblasTranspose TransA, const clblasTranspose TransB, const int m, const int n, const int k, const T alpha, const T* A, const T* x, const T beta, T* y) {

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	void*   A_base    = clGetMemoryBase(A);
	if ( ! A_base ) {
	  LOG(ERROR)<<"failed to get OpenCL device memory address for virtual address @ "<<A;
	  return false;
	}
	const void* A_device = NULL;
	if ( ! clMakeLogical(A_base, &A_device) ) {
	  return false;
	}
	size_t  A_offset  = clGetMemoryOffset(A);
	if ( A_offset < 0 ) {
	  LOG(ERROR)<<"failed to get OpenCL device memory offset for virtual address @ "<<A;
	  return false;
	}
	A_offset /= sizeof(T);
	//DLOG(INFO)<<A<<" -> "<<A_base<<" with offset = "<<A_offset;

  void*   x_base    = clGetMemoryBase(x);
  if ( ! x_base ) {
    LOG(ERROR)<<"failed to get OpenCL device memory address for virtual address @ "<<x;
    return false;
  }
  const void* x_device = NULL;
  if ( ! clMakeLogical(x_base, &x_device) ) {
    return false;
  }
  size_t  x_offset  = clGetMemoryOffset(x);
  if ( x_offset < 0 ) {
    LOG(ERROR)<<"failed to get OpenCL device memory offset for virtual address @ "<<x;
    return false;
  }
  x_offset /= sizeof(T);
  //DLOG(INFO)<<x<<" -> "<<x_base<<" with offset = "<<x_offset;

  void*   y_base    = clGetMemoryBase(y);
  if ( ! y_base ) {
    LOG(ERROR)<<"failed to get OpenCL device memory address for virtual address @ "<<y;
    return false;
  }
  const void* y_device = NULL;
  if ( ! clMakeLogical(y_base, &y_device) ) {
    return false;
  }
  size_t  y_offset  = clGetMemoryOffset(y);
  if ( y_offset < 0 ) {
    LOG(ERROR)<<"failed to get OpenCL device memory offset for virtual address @ "<<y;
    return false;
  }
  y_offset /= sizeof(T);
  //DLOG(INFO)<<y<<" -> "<<y_base<<" with offset = "<<y_offset;

  /*
	std::vector<cl_mem> sb;
	std::map<const void*, std::pair<void*, size_t> > bm;

	const void* A_logical;
	if ( ! clMakeLogical2(A, &A_logical, sb, bm) ) {
		return false;
	}

	const void* x_logical;
	if ( ! clMakeLogical2(x, &x_logical, sb, bm) ) {
		return false;
	}

	const void* y_logical;
	if ( ! clMakeLogical2(y, &y_logical, sb, bm) ) {
		return false;
	}
  */

	// Note that cublas follows fortran order.
	int lda = (TransA == clblasNoTrans) ? k : m;
	int ldb = (TransB == clblasNoTrans) ? n : k;
	int ldc = n;

	/*
	std::cout<<"TransA = "<<TransA<<std::endl;
	std::cout<<"TransB = "<<TransB<<std::endl;

	std::cout<<"m   = "<<m<<std::endl;
	std::cout<<"n   = "<<n<<std::endl;
	std::cout<<"k   = "<<k<<std::endl;

	std::cout<<"A = ["<<m<<" x "<<k<<"]"<<std::endl;
	std::cout<<"x = ["<<k<<" x "<<n<<"]"<<std::endl;
	std::cout<<"y = ["<<m<<" x "<<n<<"]"<<std::endl;

	std::cout<<"lda = "<<lda<<std::endl;
	std::cout<<"ldb = "<<ldb<<std::endl;
	std::cout<<"ldc = "<<ldc<<std::endl;
	*/

	if( typeid(T) == typeid(float) ) {
		if ( ! CL_CHECK( clblasSgemm(clblasRowMajor, TransA, TransB, m, n, k, alpha, (cl_mem) A_device, A_offset, lda, (cl_mem) x_device, x_offset, ldb, beta, (cl_mem) y_device, y_offset, ldc, 1, queue, 0, NULL, NULL) ) ) {
			LOG(ERROR) << "clblasSgemm() failed on GPU "<<gpu->name();
			//clReleaseSubBuffers(sb);
			//clReleaseBufferMap(bm);
			return false;
		}
		////clFinish(*queue);
		DLOG(INFO) << "clblasSgemm() succeeded on GPU "<<gpu->name();
	}

	if( typeid(T) == typeid(double) ) {
		if ( ! CL_CHECK( clblasDgemm(clblasRowMajor, TransA, TransB, m, n, k, alpha, (cl_mem) A_device, A_offset, lda, (cl_mem) x_device, x_offset, ldb, beta, (cl_mem) y_device, y_offset, ldc, 1, queue, 0, NULL, NULL) ) ) {
			LOG(ERROR) << "clblasDgemm() failed on GPU "<<gpu->name();
			return false;
			//clReleaseSubBuffers(sb);
			//clReleaseBufferMap(bm);
		}
		////clFinish(*queue);
		DLOG(INFO) << "clblasDgemm() succeeded on GPU "<<gpu->name();
	}

	/*
	clReleaseBufferMap(bm);
	clReleaseSubBuffers(sb);
	*/
	return true;
}
template bool clBLASgemm<float>(const clblasTranspose TransA, const clblasTranspose TransB, const int m, const int n, const int k, const float alpha, const float* A, const float* x, const float beta, float* y);
template bool clBLASgemm<double>(const clblasTranspose TransA, const clblasTranspose TransB, const int m, const int n, const int k, const double alpha, const double* A, const double* x, const double beta, double* y);

template<typename T>
bool clBLASgemm(const clblasTranspose TransA, const clblasTranspose TransB, const int m, const int n, const int k, const T alpha, const T* A, const int step_A, const T* x, const int step_x, const T beta, T* y, const int step_y) {

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	std::vector<cl_mem> sb;
	std::map<const void*, std::pair<void*, size_t> > bm;

	const void* A_logical;
	if ( ! clMakeLogical2(A, &A_logical, sb, bm) ) {
		return false;
	}

	const void* x_logical;
	if ( ! clMakeLogical2(x, &x_logical, sb, bm) ) {
		return false;
	}

	const void* y_logical;
	if ( ! clMakeLogical2(y, &y_logical, sb, bm) ) {
		return false;
	}

	// Note that cublas follows fortran order.
	int lda = (TransA == clblasNoTrans) ? k : m;
	int ldb = (TransB == clblasNoTrans) ? n : k;
	int ldc = n;

	//TIME("clblasGemm()",
	  //  {
	if( typeid(T) == typeid(float) ) {
		if ( ! CL_CHECK( clblasSgemm(clblasRowMajor, TransA, TransB, m, n, k, alpha, (cl_mem) A_logical, step_A, lda, (cl_mem) x_logical, step_x, ldb, beta, (cl_mem) y_logical, step_y, ldc, 1, queue, 0, NULL, NULL) ) ) {
			LOG(ERROR) << "clblasSgemm() failed on GPU "<<gpu->name();
			return false;
		}
		////clFinish(*queue);
		DLOG(INFO) << "clblasSgemm() succeeded on GPU "<<gpu->name();
	}

	if( typeid(T) == typeid(double) ) {
		if ( ! CL_CHECK( clblasDgemm(clblasRowMajor, TransA, TransB, m, n, k, alpha, (cl_mem) A_logical, step_A, lda, (cl_mem) x_logical, step_x, ldb, beta, (cl_mem) y_logical, step_y, ldc, 1, queue, 0, NULL, NULL) ) ) {
			LOG(ERROR) << "clblasDgemm() failed on GPU "<<gpu->name();
			return false;
		}
		////clFinish(*queue);
		DLOG(INFO) << "clblasDgemm() succeeded on GPU "<<gpu->name();
	}
	    //});

	clReleaseBufferMap(bm);
	clReleaseSubBuffers(sb);

	return true;
}
template bool clBLASgemm<float>(const clblasTranspose TransA, const clblasTranspose TransB, const int m, const int n, const int k, const float alpha, const float* A, const int step_A, const float* x, const int step_x, const float beta, float* y, const int step_y);
template bool clBLASgemm<double>(const clblasTranspose TransA, const clblasTranspose TransB, const int m, const int n, const int k, const double alpha, const double* A, const int step_A, const double* x, const int step_x, const double beta, double* y, const int step_y);


template<typename T>
bool clBLASaxpy(const int N, const T alpha, const T* X, const int incr_x, T* Y, const int incr_y) {

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	std::vector<cl_mem> sb;
	std::map<const void*, std::pair<void*, size_t> > bm;

	const void* X_logical;
	if ( ! clMakeLogical2(X, &X_logical, sb, bm) ) {
		return false;
	}

	const void* Y_logical;
	if ( ! clMakeLogical2(Y, &Y_logical, sb, bm) ) {
		return false;
	}

	if( typeid(T) == typeid(float) ) {

		if ( ! CL_CHECK( clblasSaxpy(N, alpha, (cl_mem) X_logical, 0, incr_x, (cl_mem) Y_logical, 0, incr_y, 1, queue, 0, NULL, NULL) ) ) {
			LOG(ERROR) << "clblasSaxpy() failed on GPU "<<gpu->name();
			clReleaseSubBuffers(sb);
			clReleaseBufferMap(bm);
			return false;
		}
		DLOG(INFO) << "clblasSaxpy() succeeded on GPU "<<gpu->name();
	}

	if( typeid(T) == typeid(double) ) {
		if ( ! CL_CHECK( clblasDaxpy(N, alpha, (cl_mem) X_logical, 0, incr_x, (cl_mem) Y_logical, 0, incr_y, 1, queue, 0, NULL, NULL) ) ) {
			LOG(ERROR) << "clblasDaxpy() failed on GPU "<<gpu->name();
			clReleaseSubBuffers(sb);
			clReleaseBufferMap(bm);
			return false;
		}
		DLOG(INFO) << "clblasDaxpy() succeeded on GPU "<<gpu->name();
	}

	clReleaseSubBuffers(sb);
	clReleaseBufferMap(bm);
	return true;
}
template bool clBLASaxpy<float>(const int N, const float alpha, const float* X, const int incr_x, float* Y, const int incr_y);
template bool clBLASaxpy<double>(const int N, const double alpha, const double* X, const int incr_x, double* Y, const int incr_y);

bool clIsVirtualMemory(const void* p) {

	return caffe::OpenCLMemory::isHighMem(p);
}

bool clMakeLogical(const void* ptr_virtual, const void** ptr_logical) {

	if ( ! clIsVirtualMemory(ptr_virtual) ) {
		LOG(WARNING) << "PTR@"<<ptr_virtual<<" is not in virtual memory.";
		*ptr_logical = ptr_virtual;
		return true;
	}

	OpenCLMemory clMem;
	if ( ! gpu->get(ptr_virtual, clMem) ) {
		LOG(ERROR) << "failed to get OpenCLMemory object associated with VM@" << ptr_virtual;
		return false;
	}
	*ptr_logical = clMem.getLogicalPointer();

	if ( clIsVirtualMemory(*ptr_logical) ) {
		DLOG(INFO) << "failed to convert VM@"<<ptr_virtual<<" from virtual to logical address space.";
		return false;
	}

	return true;
}

bool clMakeLogical2(const void* ptr_virtual, const void** ptr_logical, std::vector<cl_mem>& subBuffers, std::map<const void*, std::pair<void*, size_t> >& bufferMap) {

	if ( ! clIsVirtualMemory(ptr_virtual) ) {
		LOG(WARNING) << "PTR@"<<ptr_virtual<<" is not in virtual memory.";
		*ptr_logical = ptr_virtual;
		return true;
	}

	OpenCLMemory clMem;
	if ( ! gpu->get(ptr_virtual, clMem) ) {
		LOG(ERROR) << "failed to get OpenCLMemory object associated with VM@" << ptr_virtual;
		return false;
	}
	*ptr_logical = clMem.getLogicalPointer();

	if ( clIsVirtualMemory(*ptr_logical) ) {
		LOG(ERROR) << "failed to convert VM@"<<ptr_virtual<<" from virtual to logical address space.";
		return false;
	}

	size_t offset 	= clGetMemoryOffset(ptr_virtual);
	size_t size 	= clGetMemorySize(ptr_virtual);
	cl_int err;
	if( offset > 0 ) {
	  LOG(INFO)<<"offset = "<<offset;

		/* check alignment */
		cl_uint bytes = gpu->getDeviceMemBaseAddrAlign();
		if ( offset % bytes != 0 ) {
			LOG(WARNING)<<"sub-buffer memory offset ("<<offset<<" Byte) is not aligned with device memory offset ("<<bytes<<" Byte)";

			cl_mem_flags flags;
			err = clGetMemObjectInfo((cl_mem) *ptr_logical, CL_MEM_FLAGS, sizeof(flags), &flags, NULL);
			if ( err != CL_SUCCESS ) {
				LOG(ERROR)<<"failed to query memory properties.";
				return false;
			}

			void* ptr_buffer;
			size_t buffer_size = size-offset;
			if ( ! clMalloc(&ptr_buffer, buffer_size) ) {
				return false;
			}

			if ( ! clMemcpy(ptr_buffer, ptr_virtual, buffer_size, caffe::OpenCL::COPY_DEFAULT) ) {
				return false;
			}

			if ( ! clMakeLogical(ptr_buffer, ptr_logical) ) {
				return false;
			}
			bufferMap[ptr_virtual] = std::make_pair(ptr_buffer, buffer_size);
			return true;
		}

		LOG(INFO)<<"memory offset = "<<offset<<" detected, create sub-buffer ["<<offset<<"|"<<size-offset<<"]";
		cl_buffer_region region = {offset, size - offset};
		*ptr_logical = (const void*) clCreateSubBuffer((cl_mem) *ptr_logical, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
		if ( err != CL_SUCCESS ) {
			LOG(ERROR) << "failed to create sub-buffer on GPU "<<gpu->name()<<" : "<<what(err);
			return false;
		}
		DLOG(INFO)<<"create sub-buffer ";
		subBuffers.push_back((cl_mem) *ptr_logical);
	}

	return true;
}


size_t clGetMemoryOffset(const void* ptr_virtual) {

	if ( ! clIsVirtualMemory(ptr_virtual) ) {
		LOG(WARNING) << "PTR@"<<ptr_virtual<<" is not in virtual memory.";
		return -1;
	}

	OpenCLMemory clMem;
	if ( ! gpu->get(ptr_virtual, clMem) ) {
		LOG(ERROR) << "failed to get OpenCLMemory object associated with VM@" << ptr_virtual;
		return -1;
	}
	size_t mem_offset = static_cast<const char*>(ptr_virtual) - static_cast<const char*>(clMem.getVirtualPointer());

	return mem_offset;
}

size_t clGetMemorySize(const void* ptr_virtual) {

	if ( ! clIsVirtualMemory(ptr_virtual) ) {
		LOG(WARNING) << "PTR@"<<ptr_virtual<<" is not in virtual memory.";
		return -1;
	}

	OpenCLMemory clMem;
	if ( ! gpu->get(ptr_virtual, clMem) ) {
		LOG(ERROR) << "failed to get OpenCLMemory object associated with VM@" << ptr_virtual;
		return -1;
	}
	return clMem.getSize();
}

void* clGetMemoryBase(const void* ptr_virtual) {

  if ( ! clIsVirtualMemory(ptr_virtual) ) {
    LOG(WARNING) << "PTR@"<<ptr_virtual<<" is not in virtual memory.";
    return NULL;
  }

  OpenCLMemory clMem;
  if ( ! gpu->get(ptr_virtual, clMem) ) {
    LOG(ERROR) << "failed to get OpenCLMemory object associated with VM@" << ptr_virtual;
    return NULL;
  }
  return clMem.getVirtualPointer();
}

template<typename T>
bool clsub(const int n, const T* array_GPU_x, const T* array_GPU_y, T* array_GPU_z) {

	std::string kernel_name = clGetKernelName<T>("clsub");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, n)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_y)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_z)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clsub<float>(const int n, const float* array_GPU_x, const float* array_GPU_y, float* array_GPU_z);
template bool clsub<double>(const int n, const double* array_GPU_x, const double* array_GPU_y, double* array_GPU_z);

template<typename T>
bool cladd(const int n, const T* array_GPU_x, const T* array_GPU_y, T* array_GPU_z) {

	std::string kernel_name = clGetKernelName<T>("cladd");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, n)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_y)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_z)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool cladd<float>(const int n, const float* array_GPU_x, const float* array_GPU_y, float* array_GPU_z);
template bool cladd<double>(const int n, const double* array_GPU_x, const double* array_GPU_y, double* array_GPU_z);

template<typename T>
bool cladd_scalar(const int N, const T alpha, T* Y) {

	std::string kernel_name = clGetKernelName<T>("cladd_scalar");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, N)
	CL_SET_TYPE_KERNEL_ARG(T, alpha)
	CL_SET_ARRAY_KERNEL_ARG(&Y)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(N, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(N, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool cladd_scalar<float>(const int N, const float alpha, float* Y);
template bool cladd_scalar<double>(const int N, const double alpha, double* Y);

template<typename T>
bool clpowx(const int n, const T* array_GPU_x, const T alpha, T* array_GPU_z) {

	std::string kernel_name = clGetKernelName<T>("clpowx");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, n)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x)
	CL_SET_TYPE_KERNEL_ARG(T, alpha)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_z)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clpowx<float>(const int n, const float* array_GPU_x, const float alpha, float* array_GPU_z);
template bool clpowx<double>(const int n, const double* array_GPU_x, const double alpha, double* array_GPU_z);

template<typename T>
bool clexp(const int n, const T* array_GPU_x, T* array_GPU_y) {

	std::string kernel_name = clGetKernelName<T>("clexp");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, n)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_x)
	CL_SET_ARRAY_KERNEL_ARG(&array_GPU_y)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clexp<float>(const int n, const float* array_GPU_x, float* array_GPU_z);
template bool clexp<double>(const int n, const double* array_GPU_x, double* array_GPU_z);

bool cl_caffe_gpu_rng_uniform(const int n, unsigned int* r) {

	size_t bytes;

	bytes = n*sizeof(double);
	double* fBuffer = (double*) malloc(bytes);
	if ( fBuffer == NULL ) {
		LOG(ERROR)<<"failed to allocate cpu memory for buffer of "<<bytes<<" Bytes.";
		return false;
	}

	bytes = n*sizeof(unsigned int);
	unsigned int* buffer = (unsigned int*) malloc(bytes);
	if ( buffer == NULL ) {
		LOG(ERROR)<<"failed to allocate cpu memory for buffer of "<<bytes<<" Bytes.";
		return false;
	}

	caffe_rng_uniform(n, (double) 0, (double) UINT_MAX, fBuffer);
	for ( int i = 0; i < n; i++ ) {
		buffer[i] = (unsigned int) fBuffer[i];
		printf("buffer[%3d] = %u\n", i, buffer[i]);
	}

	BOOL_CHECK( caffe::OpenCL::clMemcpy(r, buffer, bytes, COPY_CPU_TO_GPU) );
	free(buffer);
	free(fBuffer);
	LOG(WARNING)<<"caffe_gpu_rng_gaussian() was executed on the CPU and random array copied back to GPU.";

	return true;
}

template<typename T>
bool cl_caffe_gpu_rng_uniform(const int n, const T a, const T b, T* r) {

	size_t	bytes = n*sizeof(T);
	T* buffer = (T*) malloc(bytes);
	if ( buffer == NULL ) {
		LOG(ERROR)<<"failed to allocate cpu memory for buffer of "<<bytes<<" Bytes.";
		return false;
	}

	caffe_rng_uniform(n, a, b, buffer);
	BOOL_CHECK( caffe::OpenCL::clMemcpy(r, buffer, bytes, COPY_CPU_TO_GPU) );
	free(buffer);
	LOG(WARNING)<<"caffe_gpu_rng_uniform() was executed on the CPU and random array copied back to GPU.";

	return true;
}
template bool cl_caffe_gpu_rng_uniform<float>(const int n, const float a, const float b, float* r);
template bool cl_caffe_gpu_rng_uniform<double>(const int n, const double a, const double b, double* r);

template<typename T>
bool cl_caffe_gpu_rng_gaussian(const int n, const T mu, const T sigma, T* r) {

	size_t	bytes = n*sizeof(T);
	T* buffer = (T*) malloc(bytes);
	if ( buffer == NULL ) {
		LOG(ERROR)<<"failed to allocate cpu memory for buffer of "<<bytes<<" Bytes.";
		return false;
	}

	caffe_rng_gaussian(n, mu, sigma, buffer);
	BOOL_CHECK( caffe::OpenCL::clMemcpy(r, buffer, bytes, COPY_CPU_TO_GPU) );
	free(buffer);
	LOG(WARNING)<<"caffe_gpu_rng_gaussian() was executed on the CPU and random array copied back to GPU.";

	return true;
}
template bool cl_caffe_gpu_rng_gaussian<float>(const int n, const float mu, const float sigma, float* r);
template bool cl_caffe_gpu_rng_gaussian<double>(const int n, const double mu, const double sigma, double* r);

template<typename T1, typename T2>
bool cl_caffe_gpu_rng_bernoulli(const int n, const T1 p, T2* r) {

	size_t	bytes = n*sizeof(T2);
	T2* buffer = (T2*) malloc(bytes);
	if ( buffer == NULL ) {
		LOG(ERROR)<<"failed to allocate cpu memory for buffer of "<<bytes<<" Bytes.";
		return false;
	}

	caffe_rng_bernoulli(n, p, buffer);
	BOOL_CHECK( caffe::OpenCL::clMemcpy(r, buffer, bytes, COPY_CPU_TO_GPU) );
	free(buffer);
	LOG(WARNING)<<"caffe_gpu_rng_bernoulli() was executed on the CPU and random array copied back to GPU.";

	return true;
}
template bool cl_caffe_gpu_rng_bernoulli<float, int>(const int n, const float p, int* r);
template bool cl_caffe_gpu_rng_bernoulli<double, int>(const int n, const double p, int* r);
template bool cl_caffe_gpu_rng_bernoulli<float, unsigned int>(const int n, const float p, unsigned int* r);
template bool cl_caffe_gpu_rng_bernoulli<double, unsigned int>(const int n, const double p, unsigned int* r);

} // namespace OpenCL

}// namespace caffee

#endif // USE_OPENCL
