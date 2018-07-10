#ifndef CAFFE_BACKEND_OPENCL_OCL_DEVICE_KERNEL_HPP_
#define CAFFE_BACKEND_OPENCL_OCL_DEVICE_KERNEL_HPP_

#include "caffe/backend/device_kernel.hpp"
#include "caffe/backend/backend.hpp"

#ifdef USE_OPENCL
#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#endif  // USE_OPENCL

namespace caffe {

class Device;

#ifdef USE_OPENCL

class OclDeviceKernel : public DeviceKernel {
 public:
  explicit OclDeviceKernel(Device *dev, string name,
                           viennacl::ocl::kernel ocl_ker, KernelArgs args);

  virtual void Execute(vector<size_t> group, vector<size_t> local);
  viennacl::ocl::kernel get_ocl_kernel();

  virtual void set_arg(uint_tp idx, const bool *arg);
  virtual void set_arg(uint_tp idx, const char *arg);
  virtual void set_arg(uint_tp idx, const int8_t *arg);
  virtual void set_arg(uint_tp idx, const int16_t *arg);
  virtual void set_arg(uint_tp idx, const int32_t *arg);
  virtual void set_arg(uint_tp idx, const int64_t *arg);
  virtual void set_arg(uint_tp idx, const uint8_t *arg);
  virtual void set_arg(uint_tp idx, const uint16_t *arg);
  virtual void set_arg(uint_tp idx, const uint32_t *arg);
  virtual void set_arg(uint_tp idx, const uint64_t *arg);
  virtual void set_arg(uint_tp idx, const half_fp *arg);
  virtual void set_arg(uint_tp idx, const float *arg);
  virtual void set_arg(uint_tp idx, const double *arg);

  virtual void set_arg(uint_tp idx, vptr<bool> *arg);
  virtual void set_arg(uint_tp idx, vptr<char> *arg);
  virtual void set_arg(uint_tp idx, vptr<int8_t> *arg);
  virtual void set_arg(uint_tp idx, vptr<int16_t> *arg);
  virtual void set_arg(uint_tp idx, vptr<int32_t> *arg);
  virtual void set_arg(uint_tp idx, vptr<int64_t> *arg);
  virtual void set_arg(uint_tp idx, vptr<uint8_t> *arg);
  virtual void set_arg(uint_tp idx, vptr<uint16_t> *arg);
  virtual void set_arg(uint_tp idx, vptr<uint32_t> *arg);
  virtual void set_arg(uint_tp idx, vptr<uint64_t> *arg);
  virtual void set_arg(uint_tp idx, vptr<half_fp> *arg);
  virtual void set_arg(uint_tp idx, vptr<float> *arg);
  virtual void set_arg(uint_tp idx, vptr<double> *arg);
  virtual void set_arg(uint_tp idx, vptr<void> *arg);

  virtual void set_arg(uint_tp idx, vptr<const bool> *arg);
  virtual void set_arg(uint_tp idx, vptr<const char> *arg);
  virtual void set_arg(uint_tp idx, vptr<const int8_t> *arg);
  virtual void set_arg(uint_tp idx, vptr<const int16_t> *arg);
  virtual void set_arg(uint_tp idx, vptr<const int32_t> *arg);
  virtual void set_arg(uint_tp idx, vptr<const int64_t> *arg);
  virtual void set_arg(uint_tp idx, vptr<const uint8_t> *arg);
  virtual void set_arg(uint_tp idx, vptr<const uint16_t> *arg);
  virtual void set_arg(uint_tp idx, vptr<const uint32_t> *arg);
  virtual void set_arg(uint_tp idx, vptr<const uint64_t> *arg);
  virtual void set_arg(uint_tp idx, vptr<const half_fp> *arg);
  virtual void set_arg(uint_tp idx, vptr<const float> *arg);
  virtual void set_arg(uint_tp idx, vptr<const double> *arg);
  virtual void set_arg(uint_tp idx, vptr<const void> *arg);

 private:
  inline void set_arg_helper(uint_tp idx, cl_mem mem, uint_tp off);

  vector<uint_tp> ocl_arg_offsets_;
  viennacl::ocl::kernel ocl_kernel_;
};

#endif  // USE_OPENCL

}


#endif  // CAFFE_BACKEND_OPENCL_OCL_DEVICE_KERNEL_HPP_
