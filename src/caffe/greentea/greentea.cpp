/*
 * greentea.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: Fabian Tschopp
 */

#include "caffe/greentea/greentea.hpp"

namespace caffe {

#ifdef USE_GREENTEA

viennacl::ocl::handle<cl_mem> WrapHandle(cl_mem in,
                                         viennacl::ocl::context *ctx) {
  if (in != NULL) {
    viennacl::ocl::handle<cl_mem> memhandle(in, *ctx);
    memhandle.inc();
    return memhandle;
  } else {
    cl_int err;
    cl_mem dummy = clCreateBuffer(ctx->handle().get(), CL_MEM_READ_WRITE, 0,
    NULL,
                                  &err);
    viennacl::ocl::handle<cl_mem> memhandle(dummy, *ctx);
    return memhandle;
  }
}

#endif

DeviceContext::DeviceContext()
    : id_(0), backend_(Backend::BACKEND_CUDA) {
}

DeviceContext::DeviceContext(int id, Backend backend)
    : id_(id), backend_(backend) {
}

Backend DeviceContext::backend() const {
  return backend_;
}

int DeviceContext::id() const {
  return id_;
}

}  // namespace caffe
