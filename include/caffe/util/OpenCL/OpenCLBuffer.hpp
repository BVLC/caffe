#ifndef __OPENCL_BUFFER_HPP__
#define __OPENCL_BUFFER_HPP__

#include <CL/cl.h>
#include <string>
#include <iostream>
#include <stdint.h>
#include <exception>
#include <caffe/util/OpenCL/OpenCLMemory.hpp>

namespace caffe {

class OpenCLBuffer : public OpenCLMemory {

public:
  OpenCLBuffer();
  OpenCLBuffer(size_t size);
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
  OpenCLBufferException(std::string message) {
    message_ = message;
  }

  virtual ~OpenCLBufferException() throw() {
  }

  virtual const char* what() const throw() {
    return message_.c_str();
  }

protected:

private:
  std::string message_;
};

} // namespace caffe

#endif // __OPENCL_BUFFER_HPP__
