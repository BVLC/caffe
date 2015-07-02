/*
 * greentea.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: Fabian Tschopp
 */

#include "caffe/common.hpp"
#include "caffe/greentea/greentea.hpp"
#include "caffe/util/device_alternate.hpp"

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


}  // namespace caffe
