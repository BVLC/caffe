#ifndef CAFFE_BACKEND_CUDA_CUDA_DEVICE_KERNEL_HPP_
#define CAFFE_BACKEND_CUDA_CUDA_DEVICE_KERNEL_HPP_

#include "caffe/backend/device_kernel.hpp"
#include "caffe/backend/backend.hpp"

namespace caffe {

#ifdef USE_CUDA

class Device;

class CudaDeviceKernel : public DeviceKernel {
 public:
  explicit CudaDeviceKernel(Device *dev, string name,
                            shared_ptr<CUfunction> cuda_kernel,
                            KernelArgs args);
  virtual void Execute(vector<size_t> group, vector<size_t> local);

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
  shared_ptr<CUfunction> cuda_kernel_;
  vector<void*> cuda_args_;

};

#endif  // USE_CUDA

}



#endif  // CAFFE_BACKEND_CUDA_CUDA_DEVICE_KERNEL_HPP_
