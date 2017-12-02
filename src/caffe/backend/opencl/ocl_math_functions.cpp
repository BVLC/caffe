#include "caffe/backend/opencl/ocl_device.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/common.hpp"
#include "caffe/backend/backend.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/dev_ptr.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"
#include "caffe/backend/opencl/ocl_dev_ptr.hpp"

namespace caffe {

#ifdef USE_OPENCL

void OclDevice::memcpy(const uint_tp n, vptr<const void> x, vptr<void> y) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());
  if (x.get_ocl_mem() != y.get_ocl_mem()) {
    clEnqueueCopyBuffer(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                        y.get_ocl_mem(), x.get_ocl_off(), y.get_ocl_off(), n, 0,
                        NULL, NULL);
  }
}

void OclDevice::memcpy(const uint_tp n, const void* x, vptr<void> y) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());
  if (x != nullptr) {
    clEnqueueWriteBuffer(ctx.get_queue().handle().get(), y.get_ocl_mem(),
                         CL_TRUE, y.get_ocl_off(), n, x, 0, NULL, NULL);
  }
}

void OclDevice::memcpy(const uint_tp n, vptr<const void> x, void* y) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());
  if (y != NULL) {
    clEnqueueReadBuffer(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                        CL_TRUE, x.get_ocl_off(), n, y, 0, NULL, NULL);
  }
}

#endif  // USE_OPENCL

}  // namespace caffe

