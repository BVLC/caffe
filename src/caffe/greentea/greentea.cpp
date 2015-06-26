/*
 * greentea.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: Fabian Tschopp
 */

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

void FinishQueues(viennacl::ocl::context *ctx) {
  for (int i = 0; i < GREENTEA_QUEUE_COUNT; ++i) {
    ctx->switch_queue(i);
    ctx->get_queue().finish();
  }
  ctx->switch_queue(0);
}

#endif


}  // namespace caffe
