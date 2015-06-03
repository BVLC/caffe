#ifndef __OPENCL_MEMORY_HPP__
#define __OPENCL_MEMORY_HPP__

#include <CL/cl.h>
#include <string>
#include <iostream>
#include <stdint.h>
#include <exception>

namespace caffe {

class OpenCLMemory {

public:
	OpenCLMemory();
	OpenCLMemory(size_t size);
	//OpenCLMemory(const OpenCLMemory& mem);
	~OpenCLMemory();

	void free();
	void* getVirtualPointer();
	const void* getLogicalPointer();
	cl_mem* getLogicalPointer2();
	std::string getTag();
	bool contains(const void* ptr);
	void print();
	bool overlaps(OpenCLMemory& clMem);
	bool includes(OpenCLMemory& clMem);
	static bool isHighMem(const void* ptr);
	size_t getSize();

	static void logStatistics() {
		DLOG(INFO)<<"numCallsMalloc = "<<numCallsMalloc;
		DLOG(INFO)<<"numCallsFree   = "<<numCallsFree;
	}

protected:

private:

	const void*		ptr_device_mem;	// the pointer to cl_mem object created by the OpenCL runtime - not a memory address
	cl_mem 		ptr_device_mem_;
	void*			ptr_virtual_bgn;
	void*			ptr_virtual_end;
	size_t 			size;
	std::string 	tag;
	static unsigned long int 	count;
	static void*	ptr_offset;
	static unsigned long int numCallsMalloc;
	static unsigned long int numCallsFree;
};

class OpenCLMemoryException: public std::exception {

public:
	OpenCLMemoryException(std::string message) {
		message_ = message;
	}

	virtual ~OpenCLMemoryException() throw() {
	}

	virtual const char* what() const throw() {
		return message_.c_str();
	}

protected:

private:
	std::string message_;
};

} // namespace caffe

#endif // __OPENCL_MEMORY_HPP__
