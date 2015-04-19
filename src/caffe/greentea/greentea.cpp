/*
 * greentea.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: Fabian Tschopp
 */

#include "caffe/greentea/greentea.hpp"

namespace caffe {

#ifdef USE_GREENTEA
template<typename Dtype>
cl_mem Subregion(cl_mem in, size_t off, size_t size) {
  cl_buffer_region* region = new cl_buffer_region();
  region->origin = sizeof(Dtype) * off;
  region->size = sizeof(Dtype) * size;
  cl_int status;
  const cl_mem out = clCreateSubBuffer(in, CL_MEM_READ_WRITE,
  CL_BUFFER_CREATE_TYPE_REGION,
                                       region, &status);
  return out;
}

template cl_mem Subregion<float>(cl_mem in, size_t off, size_t size);
template cl_mem Subregion<double>(cl_mem in, size_t off, size_t size);
template cl_mem Subregion<long>(cl_mem in, size_t off, size_t size);
template cl_mem Subregion<int>(cl_mem in, size_t off, size_t size);

viennacl::ocl::handle<cl_mem> WrapHandle(cl_mem in,
                                         viennacl::ocl::context &ctx) {
  if (in != NULL) {
    viennacl::ocl::handle<cl_mem> memhandle(in, ctx);
    memhandle.inc();
    return memhandle;
  } else {
    cl_int err;
    cl_mem dummy = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE, 0,
                                  NULL, &err);
    viennacl::ocl::handle<cl_mem> memhandle(dummy, ctx);
    return memhandle;
  }
}

#endif

DeviceContext::DeviceContext()
    : id_(0),
      backend_(Backend::BACKEND_CUDA) {

}

DeviceContext::DeviceContext(int id, Backend backend)
    : id_(id),
      backend_(backend) {

}

Backend DeviceContext::backend() const {
  return backend_;
}

int DeviceContext::id() const {
  return id_;
}

}
