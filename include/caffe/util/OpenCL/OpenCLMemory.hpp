#ifndef __OPENCL_MEMORY_HPP__
#define __OPENCL_MEMORY_HPP__

#include <CL/cl.h>
#include <glog/logging.h>
#include <stdint.h>

#include <exception>
#include <iostream>  // NOLINT(*)
#include <string>

namespace caffe {

class OpenCLMemory {
 public:
    OpenCLMemory();
    explicit OpenCLMemory(size_t size);
    ~OpenCLMemory();

    void free();
    void* getVirtualPointer();
    const void* getLogicalPointer();
    cl_mem* getLogicalPointer2();
    std::string getTag();
    bool contains(const void* ptr);
    void print();
    bool overlaps(OpenCLMemory& clMem);  // NOLINT(*)
    bool includes(OpenCLMemory& clMem);  // NOLINT(*)
    static bool isHighMem(const void* ptr);
    size_t getSize();
    bool hasEvent();
    bool setEvent(cl_event event);
    cl_event getEvent();
    void resetEvent();

    static void logStatistics() {
      DLOG(INFO) << "OpenCL Memory Statistics [clMalloc|clFree] = ["
                 << numCallsMalloc << "|" << numCallsFree << "]";
    }

 private:
    OpenCLMemory(const OpenCLMemory&);

    /* the pointer to cl_mem object created by the OpenCL runtime
       this is not a memory address
       */
    const void* ptr_device_mem;
    cl_mem ptr_device_mem_;
    void* ptr_virtual_bgn;
    void* ptr_virtual_end;
    size_t size;
    std::string tag;
    static uint64_t count;
    static void* ptr_offset;
    static uint64_t numCallsMalloc;
    static uint64_t numCallsFree;
    cl_event memoryEvent;
};

class OpenCLMemoryException: public std::exception {
 public:
    explicit OpenCLMemoryException(std::string message) {
      message_ = message;
    }

    virtual ~OpenCLMemoryException() throw () {  // NOLINT(*)
    }

    virtual const char* what() const throw () {  // NOLINT(*)
      return message_.c_str();
    }

 protected:
 private:
    std::string message_;
};

}  // namespace caffe

#endif  // __OPENCL_MEMORY_HPP__
