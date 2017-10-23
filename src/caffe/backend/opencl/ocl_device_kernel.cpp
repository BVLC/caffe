#include <algorithm>

#include "caffe/backend/opencl/ocl_device_kernel.hpp"
#include "caffe/common.hpp"
#include "caffe/backend/device.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"

namespace caffe {

#ifdef USE_OPENCL

OclDeviceKernel::OclDeviceKernel(Device* dev, viennacl::ocl::kernel ocl_ker,
                                     KernelArgs args) {
  this->device_ = dev;
  this->ocl_kernel_ = ocl_ker;
  this->args_ = args;
}

void OclDeviceKernel::Execute(vector<size_t> group,
                                vector<size_t> local) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->device_->id());

  cl_kernel kernel = ocl_kernel_.handle().get();

  uint_tp work_dim = std::min(local.size(), group.size());

  vector<size_t> local_ws;
  vector<size_t> global_ws;

  for (int_tp i = 0; i < work_dim; ++i) {
    global_ws.push_back(local[i] * group[i]);
    local_ws.push_back(local[i]);
  }

  const size_t *global_ws_ptr = &global_ws[0];
  const size_t *local_ws_ptr = &local_ws[0];

  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                   kernel, work_dim, NULL,
                                   global_ws_ptr, local_ws_ptr,
                                   0, NULL, NULL));

  // Reset kernel arguments
  this->arg_idx_ = 0;
}

viennacl::ocl::kernel OclDeviceKernel::get_ocl_kernel() {
  return ocl_kernel_;
}

void OclDeviceKernel::set_arg(uint_tp idx, const bool *arg) {
  int8_t converted_arg = static_cast<int8_t>(*arg);
  clSetKernelArg(this->ocl_kernel_.handle().get(), idx,
                 safe_sizeof<int8_t>(), &converted_arg);
  this->arg_idx_ = std::max(idx, this->arg_idx_) + 1;
}
void OclDeviceKernel::set_arg(uint_tp idx, const char *arg) {
  clSetKernelArg(this->ocl_kernel_.handle().get(), idx,
                 safe_sizeof<char>(), &arg);
  this->arg_idx_ = std::max(idx, this->arg_idx_) + 1;
}
void OclDeviceKernel::set_arg(uint_tp idx, const int8_t *arg) {
  clSetKernelArg(this->ocl_kernel_.handle().get(), idx,
                 safe_sizeof<int8_t>(), arg);
  this->arg_idx_ = std::max(idx, this->arg_idx_) + 1;
}
void OclDeviceKernel::set_arg(uint_tp idx, const int16_t *arg) {
  clSetKernelArg(this->ocl_kernel_.handle().get(), this->arg_idx_,
                 safe_sizeof<int16_t>(), arg);
  this->arg_idx_ = std::max(idx, this->arg_idx_) + 1;
}
void OclDeviceKernel::set_arg(uint_tp idx, const int32_t  *arg) {
  clSetKernelArg(this->ocl_kernel_.handle().get(), this->arg_idx_,
                 safe_sizeof<int32_t>(), arg);
  this->arg_idx_ = std::max(idx, this->arg_idx_) + 1;
}
void OclDeviceKernel::set_arg(uint_tp idx, const int64_t *arg) {
  clSetKernelArg(this->ocl_kernel_.handle().get(), this->arg_idx_,
                 safe_sizeof<int64_t>(), arg);
  this->arg_idx_ = std::max(idx, this->arg_idx_) + 1;
}
void OclDeviceKernel::set_arg(uint_tp idx, const uint8_t *arg) {
  clSetKernelArg(this->ocl_kernel_.handle().get(), this->arg_idx_,
                 safe_sizeof<uint8_t>(), arg);
  this->arg_idx_ = std::max(idx, this->arg_idx_) + 1;
}
void OclDeviceKernel::set_arg(uint_tp idx, const uint16_t *arg) {
  clSetKernelArg(this->ocl_kernel_.handle().get(), this->arg_idx_,
                 safe_sizeof<uint16_t>(), arg);
  this->arg_idx_ = std::max(idx, this->arg_idx_) + 1;
}
void OclDeviceKernel::set_arg(uint_tp idx, const uint32_t *arg) {
  clSetKernelArg(this->ocl_kernel_.handle().get(), this->arg_idx_,
                 safe_sizeof<uint32_t>(), arg);
  this->arg_idx_ = std::max(idx, this->arg_idx_) + 1;
}
void OclDeviceKernel::set_arg(uint_tp idx, const uint64_t *arg) {
  clSetKernelArg(this->ocl_kernel_.handle().get(), this->arg_idx_,
                 safe_sizeof<uint64_t>(), arg);
  this->arg_idx_ = std::max(idx, this->arg_idx_) + 1;
}
void OclDeviceKernel::set_arg(uint_tp idx, const half_float::half *arg) {
  float converted_arg = static_cast<float>(*arg);
  clSetKernelArg(this->ocl_kernel_.handle().get(), this->arg_idx_,
                 safe_sizeof<float>(), &converted_arg);
  this->arg_idx_ = std::max(idx, this->arg_idx_) + 1;
}
void OclDeviceKernel::set_arg(uint_tp idx, const float *arg) {
  clSetKernelArg(this->ocl_kernel_.handle().get(), this->arg_idx_,
                 safe_sizeof<float>(), arg);
  this->arg_idx_ = std::max(idx, this->arg_idx_) + 1;
}
void OclDeviceKernel::set_arg(uint_tp idx, const double *arg) {
  clSetKernelArg(this->ocl_kernel_.handle().get(), this->arg_idx_,
                 safe_sizeof<double>(), arg);
  this->arg_idx_ = std::max(idx, this->arg_idx_) + 1;
}

inline void OclDeviceKernel::set_arg_helper(uint_tp idx, cl_mem mem,
                                                   uint_tp off) {
  clSetKernelArg(this->ocl_kernel_.handle().get(), this->arg_idx_,
                 sizeof(cl_mem), &mem);
  this->arg_idx_ = std::max(idx, this->arg_idx_) + 1;
  uint64_t flags = args_.size() > idx ? std::get<2>(args_[idx]) : 0ULL;
  if ((flags & KERNEL_ARG_MEM_OFFSET) == KERNEL_ARG_MEM_OFFSET) {
    clSetKernelArg(this->ocl_kernel_.handle().get(), this->arg_idx_,
                   safe_sizeof<uint_tp>(), &off);
    this->arg_idx_ = std::max(idx, this->arg_idx_) + 1;
  }
}

void OclDeviceKernel::set_arg(uint_tp idx, vptr<int8_t> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<int16_t> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<int32_t> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<int64_t> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<uint8_t> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<uint16_t> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<uint32_t> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<uint64_t> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<half_float::half> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<float> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<double> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<void> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}

void OclDeviceKernel::set_arg(uint_tp idx, vptr<const int8_t> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<const int16_t> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<const int32_t> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<const int64_t> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<const uint8_t> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<const uint16_t> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<const uint32_t> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<const uint64_t> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<const half_float::half> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<const float> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<const double> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<const void> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}

#endif  // USE_OPENCL

}
