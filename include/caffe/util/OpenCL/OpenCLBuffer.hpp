#ifndef __OPENCL_BUFFER_HPP__
#define __OPENCL_BUFFER_HPP__

#include <CL/cl.h>
#include <stdint.h>

#include <caffe/util/OpenCL/OpenCLMemory.hpp>

#include <exception>
#include <iostream>  // NOLINT(*)
#include <string>

namespace caffe {

class OpenCLBuffer: public OpenCLMemory {
 public:
  OpenCLBuffer();
  explicit OpenCLBuffer(size_t size);
  ~OpenCLBuffer();

  bool isAvailable();
  void setAvailable();
  void setUnavailable();

 protected:
 private:
  OpenCLBuffer(const OpenCLBuffer&);
  bool available_;
};

class OpenCLBufferException: public std::exception {
 public:
  explicit OpenCLBufferException(std::string message) {
    message_ = message;
  }

  virtual ~OpenCLBufferException() throw () {  // NOLINT(*)
  }

  virtual const char* what() const throw () {  // NOLINT(*)
    return message_.c_str();
  }

 protected:
 private:
  std::string message_;
};

}  // namespace caffe

#endif  // __OPENCL_BUFFER_HPP__
