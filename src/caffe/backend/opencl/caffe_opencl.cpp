#include <string>
#include "caffe/common.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"

namespace caffe {

#ifdef USE_OPENCL

viennacl::ocl::handle<cl_mem> WrapHandle(cl_mem in,
                                         viennacl::ocl::context *ctx) {
  if (in != nullptr) {
    // Valid cl_mem object, wrap to ViennaCL and return handle.
    viennacl::ocl::handle<cl_mem> memhandle(in, *ctx);
    memhandle.inc();
    return memhandle;
  } else {
    // Trick to pass nullptr via ViennaCL into OpenCL kernels.
    viennacl::ocl::handle<cl_mem> memhandle;
    return memhandle;
  }
}

bool IsBeignet(viennacl::ocl::context *ctx) {
  return ctx->devices()[0].opencl_c_version().find("beignet")
         != std::string::npos;
}

#endif


}  // namespace caffe
