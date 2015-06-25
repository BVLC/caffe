#ifdef USE_OPENCL

#include <CL/cl.h>
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <glog/logging.h>
#include <caffe/util/OpenCL/OpenCLMemory.hpp>
#include <caffe/util/OpenCL/OpenCLSupport.hpp>
#include <exception>

#define CL_PTR_BIT 63
namespace caffe {

long unsigned int caffe::OpenCLMemory::count = 0;
long unsigned int caffe::OpenCLMemory::numCallsMalloc = 0;
long unsigned int caffe::OpenCLMemory::numCallsFree = 0;

void* caffe::OpenCLMemory::ptr_offset = reinterpret_cast<void*>(1UL << CL_PTR_BIT);

OpenCLMemory::OpenCLMemory() {

	this->ptr_device_mem 		= NULL;
	this->ptr_device_mem_ 	= NULL;
	this->ptr_virtual_bgn 	= NULL;
	this->ptr_virtual_end 	= NULL;
	this->size 							= 0;
	this->tag								= "";
	this->memoryEvent       = NULL;
}

OpenCLMemory::OpenCLMemory(size_t size) {

  OpenCLDevice& current_device  = OpenCLManager::CurrentPlatform()->CurrentDevice();
  cl_context context            = current_device.getContext();

  if ( ! context ) {
		std::ostringstream oss;
    oss << current_device.name() << "> failed to get OpenCL context.";
    LOG(FATAL) << oss;
	}

	double allocSizeMB = size / (1024.0*1024.0);

	cl_int err;
  this->ptr_device_mem_ = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
	if ( err != CL_SUCCESS ) {
		std::ostringstream oss;
    oss << current_device.name() << "> failed to create CL_MEM_READ_WRITE buffer of "<<allocSizeMB<<" MByte";
    LOG(FATAL)<<oss.str();
	}
  size_t bytesUsed = current_device.getMemoryUsage();
	double deviceMemUsedMB = bytesUsed;
	deviceMemUsedMB /= 1024.0;
	deviceMemUsedMB /= 1024.0;

	this->ptr_device_mem 		= (const void*) this->ptr_device_mem_;
	this->size 							= size;
	this->ptr_virtual_bgn		= ptr_offset;
	this->ptr_virtual_end		= static_cast<void*>((char*) ptr_offset + size -1 );
	this->ptr_offset				= static_cast<void*>((char*) ptr_offset + size);
	this->memoryEvent       = NULL;

	std::ostringstream oss;
	oss << "GPU@" << this->ptr_device_mem << "("<<this->size<<") MEM"<< this->count;
	this->tag = oss.str();

  DLOG(INFO) << current_device.name() << "> create CL_MEM_READ_WRITE buffer of "<<allocSizeMB<<" MByte at "<<getTag().c_str()<<" total mem utilization = "<<deviceMemUsedMB<<" MByte";
	DLOG(INFO) << "new memory "<<this->tag.c_str();
	this->count++;
	numCallsMalloc++;
	logStatistics();
}

void OpenCLMemory::free() {

  OpenCLDevice& current_device = OpenCLManager::CurrentPlatform()->CurrentDevice();
  cl_int err;

	if ( this->ptr_device_mem_ != NULL ) {
	  if ( this->hasEvent() ) {
	    CL_CHECK( clWaitForEvents(1, &this->memoryEvent) );
	    this->resetEvent();
	  }
		err = clReleaseMemObject(this->ptr_device_mem_);
		if ( err != CL_SUCCESS ) {
				std::ostringstream oss;
        oss << current_device.name() << "> failed to call clReleaseMemObject("<<this->ptr_device_mem<<").";
				LOG(ERROR) << oss.str().c_str();
				throw OpenCLMemoryException(oss.str());
		}
		this->ptr_device_mem_ = NULL;
    DLOG(INFO) << current_device.name() << "> clReleaseMemObject(" << this->ptr_device_mem<< ") succeeded.";
		numCallsFree++;
		logStatistics();
	}
}

OpenCLMemory::~OpenCLMemory() {
}

bool OpenCLMemory::isHighMem(const void* p) {

	if ((uintptr_t) p & (1UL << CL_PTR_BIT)) {
		return true;
	}
	return false;
}

void* OpenCLMemory::getVirtualPointer() {
	return ptr_virtual_bgn;
}

const void* OpenCLMemory::getLogicalPointer() {
	return ptr_device_mem;
}

cl_mem* OpenCLMemory::getLogicalPointer2() {
	return &ptr_device_mem_;
}

std::string OpenCLMemory::getTag() {
	return tag;
}

bool OpenCLMemory::contains(const void* ptr) {

	if ( ptr == NULL ) {
		LOG(ERROR) << "input pointer == NULL";
		return false;
	}

	if ( this->ptr_device_mem == NULL ) {
		LOG(ERROR) << "memory pointer == NULL";
		return false;
	}

	if ( ! caffe::OpenCLMemory::isHighMem(ptr) ) {
		LOG(ERROR) << "not a virtual memory pointer @"<<ptr;
		return false;
	}

	if ( ptr < this->ptr_virtual_bgn ) {
		return false;
	}
	if ( ptr > this->ptr_virtual_end ) {
		return false;
	}

	return true;
}

void OpenCLMemory::print() {

	LOG(INFO) << this->getTag().c_str() << " size = " << size;
}

bool OpenCLMemory::overlaps(OpenCLMemory& clMem) {

	if ( this->ptr_device_mem == NULL ) {
		return false;
	}
	if ( clMem.ptr_device_mem == NULL ) {
		return false;
	}
	if ( this->ptr_virtual_end < clMem.ptr_virtual_bgn ) {
		return false;
	}
	if ( clMem.ptr_virtual_end < this->ptr_virtual_bgn ) {
		return false;
	}
	return true;
}

bool OpenCLMemory::includes(OpenCLMemory& clMem) {

	if ( this->ptr_device_mem == NULL ) {
		return false;
	}
	if ( clMem.ptr_device_mem == NULL ) {
		return false;
	}
	if ( this->ptr_virtual_bgn <= clMem.ptr_virtual_bgn && this->ptr_virtual_end >= clMem.ptr_virtual_bgn ) {
		return true;
	}
	return false;
}

size_t OpenCLMemory::getSize() {

	return this->size;
}

bool OpenCLMemory::hasEvent() {
  if ( this->memoryEvent != NULL ) {
    return true;
  } else {
    return false;
  }
}

bool OpenCLMemory::setEvent(cl_event event) {
  DLOG(INFO)<<"setting event for "<<this->getTag()<<" event = "<<event;
  this->memoryEvent = event;
  return true;
}

cl_event OpenCLMemory::getEvent() {
  if ( this->memoryEvent != NULL ) {
    DLOG(INFO)<<"get event for "<<this->getTag()<<" event = "<<this->memoryEvent;
    return this->memoryEvent;
  } else {
    return NULL;
  }
}

void OpenCLMemory::resetEvent() {
  this->memoryEvent = NULL;
}

} // namespace caffe

#endif // USE_OPENCL
