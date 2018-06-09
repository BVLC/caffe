#include <algorithm>

#include "caffe/backend/opencl/ocl_device_kernel.hpp"
#include "caffe/common.hpp"
#include "caffe/backend/device.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"
#include "caffe/util/type_utils.hpp"

namespace caffe {

#ifdef USE_OPENCL

OclDeviceKernel::OclDeviceKernel(Device* dev,
                                 string name,
                                 viennacl::ocl::kernel ocl_ker,
                                 KernelArgs args) :
                                     DeviceKernel(dev, name, args),
                                 ocl_arg_offsets_(args.size()),
                                 ocl_kernel_(ocl_ker) {
  int_tp offset = 0;
  for (size_t i = 0; i < args.size(); ++i) {
    ocl_arg_offsets_[i] = offset;
    uint64_t flags = std::get<2>(args[i]);
    if ((flags & KERNEL_ARG_MEM_OFFSET) == KERNEL_ARG_MEM_OFFSET) {
      ++offset;
    }
  }
}

void OclDeviceKernel::Execute(vector<size_t> group,
                              vector<size_t> local) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->device_->id());

  cl_kernel kernel = ocl_kernel_.handle().get();

  uint_tp work_dim = std::min(local.size(), group.size());

  vector<size_t> local_ws;
  vector<size_t> global_ws;

  // Flag if OpenCL should determine the local work size instead
  bool auto_local_ws = false;

  for (int_tp i = 0; i < work_dim; ++i) {
    auto_local_ws = auto_local_ws || (local[i] == 0);
    if (auto_local_ws) {
      global_ws.push_back(group[i]);
    } else {
      global_ws.push_back(local[i] * group[i]);
    }
    local_ws.push_back(local[i]);
  }

  const size_t *global_ws_ptr = &global_ws[0];
  const size_t *local_ws_ptr = &local_ws[0];

  if (auto_local_ws) {
    local_ws_ptr = NULL;
  }


  cl_event event;

  OCL_CHECK_MESSAGE(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                           kernel, work_dim, NULL,
                                           global_ws_ptr, local_ws_ptr,
                                           0, NULL, &event), this->name_);

  // Verbose kernel debugging
#ifndef NDEBUG
  cl_int error;
  cl_int status;
  error = clWaitForEvents(1, &event);
  if (error != CL_SUCCESS) {
    error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
     OCL_CHECK_MESSAGE(error, this->name_);
     if(status != CL_COMPLETE ) {
       LOG(FATAL) << "Error executing the kernel " << status
                  << " (" << this->name_ << ")";
    }
    for (int_tp i = 0; i < work_dim; ++i) {
      std::cout << "Global: " << global_ws[i] << std::endl;
      std::cout << "Local: " << local_ws[i] << std::endl;
    }
  }
#endif  // NDEBUG

  // Reset kernel arguments
  this->arg_idx_ = 0;
}

viennacl::ocl::kernel OclDeviceKernel::get_ocl_kernel() {
  return ocl_kernel_;
}

void OclDeviceKernel::set_arg(uint_tp idx, const bool *arg) {
  int8_t converted_arg = *arg ? 1 : 0;
  int_tp curr_arg_idx = idx + ocl_arg_offsets_[idx];
  clSetKernelArg(this->ocl_kernel_.handle().get(), curr_arg_idx,
                 safe_sizeof<int8_t>(), &converted_arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void OclDeviceKernel::set_arg(uint_tp idx, const char *arg) {
  int_tp curr_arg_idx = idx + ocl_arg_offsets_[idx];
  clSetKernelArg(this->ocl_kernel_.handle().get(), curr_arg_idx,
                 safe_sizeof<char>(), arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void OclDeviceKernel::set_arg(uint_tp idx, const int8_t *arg) {
  int_tp curr_arg_idx = idx + ocl_arg_offsets_[idx];
  clSetKernelArg(this->ocl_kernel_.handle().get(), curr_arg_idx,
                 safe_sizeof<int8_t>(), arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void OclDeviceKernel::set_arg(uint_tp idx, const int16_t *arg) {
  int_tp curr_arg_idx = idx + ocl_arg_offsets_[idx];
  clSetKernelArg(this->ocl_kernel_.handle().get(), curr_arg_idx,
                 safe_sizeof<int16_t>(), arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void OclDeviceKernel::set_arg(uint_tp idx, const int32_t  *arg) {
  int_tp curr_arg_idx = idx + ocl_arg_offsets_[idx];
  clSetKernelArg(this->ocl_kernel_.handle().get(), curr_arg_idx,
                 safe_sizeof<int32_t>(), arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void OclDeviceKernel::set_arg(uint_tp idx, const int64_t *arg) {
  int_tp curr_arg_idx = idx + ocl_arg_offsets_[idx];
  clSetKernelArg(this->ocl_kernel_.handle().get(), curr_arg_idx,
                 safe_sizeof<int64_t>(), arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void OclDeviceKernel::set_arg(uint_tp idx, const uint8_t *arg) {
  int_tp curr_arg_idx = idx + ocl_arg_offsets_[idx];
  clSetKernelArg(this->ocl_kernel_.handle().get(), curr_arg_idx,
                 safe_sizeof<uint8_t>(), arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void OclDeviceKernel::set_arg(uint_tp idx, const uint16_t *arg) {
  int_tp curr_arg_idx = idx + ocl_arg_offsets_[idx];
  clSetKernelArg(this->ocl_kernel_.handle().get(), curr_arg_idx,
                 safe_sizeof<uint16_t>(), arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void OclDeviceKernel::set_arg(uint_tp idx, const uint32_t *arg) {
  int_tp curr_arg_idx = idx + ocl_arg_offsets_[idx];
  clSetKernelArg(this->ocl_kernel_.handle().get(), curr_arg_idx,
                 safe_sizeof<uint32_t>(), arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void OclDeviceKernel::set_arg(uint_tp idx, const uint64_t *arg) {
  int_tp curr_arg_idx = idx + ocl_arg_offsets_[idx];
  clSetKernelArg(this->ocl_kernel_.handle().get(), curr_arg_idx,
                 safe_sizeof<uint64_t>(), arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void OclDeviceKernel::set_arg(uint_tp idx, const half_fp *arg) {
  float converted_arg = static_cast<float>(*arg);
  int_tp curr_arg_idx = idx + ocl_arg_offsets_[idx];
  clSetKernelArg(this->ocl_kernel_.handle().get(), curr_arg_idx,
                 safe_sizeof<float>(), &converted_arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void OclDeviceKernel::set_arg(uint_tp idx, const float *arg) {
  int_tp curr_arg_idx = idx + ocl_arg_offsets_[idx];
  clSetKernelArg(this->ocl_kernel_.handle().get(), curr_arg_idx,
                 safe_sizeof<float>(), arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void OclDeviceKernel::set_arg(uint_tp idx, const double *arg) {
  int_tp curr_arg_idx = idx + ocl_arg_offsets_[idx];
  clSetKernelArg(this->ocl_kernel_.handle().get(), curr_arg_idx,
                 safe_sizeof<double>(), arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}

inline void OclDeviceKernel::set_arg_helper(uint_tp idx, cl_mem mem,
                                            uint_tp off) {
  int_tp curr_arg_idx = idx + ocl_arg_offsets_[idx];
  clSetKernelArg(this->ocl_kernel_.handle().get(), curr_arg_idx,
                 sizeof(cl_mem), &mem);
  uint64_t flags = args_.size() > idx ? std::get<2>(args_[idx]) : 0ULL;
  if ((flags & KERNEL_ARG_MEM_OFFSET) == KERNEL_ARG_MEM_OFFSET) {
    clSetKernelArg(this->ocl_kernel_.handle().get(), curr_arg_idx + 1,
                   safe_sizeof<uint_tp>(), &off);
  } else {
    // Require kernel to either have support for memory offset, or
    // keep the virtual pointer offset at 0 to avoid side effects.
    CHECK(off == 0) << "Kernel does not support memory offset, but "
                    << "the virtual pointer has a non-zero offset.";
  }
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}

void OclDeviceKernel::set_arg(uint_tp idx, vptr<bool> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<char> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
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
void OclDeviceKernel::set_arg(uint_tp idx, vptr<half_fp> *arg) {
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

void OclDeviceKernel::set_arg(uint_tp idx, vptr<const bool> *arg) {
  set_arg_helper(idx, arg->get_ocl_mem(), arg->get_ocl_off());
}
void OclDeviceKernel::set_arg(uint_tp idx, vptr<const char> *arg) {
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
void OclDeviceKernel::set_arg(uint_tp idx, vptr<const half_fp> *arg) {
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
