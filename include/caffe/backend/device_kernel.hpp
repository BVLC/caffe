#ifndef CAFFE_BACKEND_DEVICE_KERNEL_HPP_
#define CAFFE_BACKEND_DEVICE_KERNEL_HPP_

#include "caffe/common.hpp"
#include "caffe/backend/backend.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/util/half_fp.hpp"

namespace caffe {

class Device;

class DeviceKernel {
 public:
  virtual void Execute(vector<size_t> local,
                       vector<size_t> group) = 0;

  void add_arg(const bool *arg);
  void add_arg(const char *arg);
  void add_arg(const int8_t *arg);
  void add_arg(const int16_t *arg);
  void add_arg(const int32_t *arg);
  void add_arg(const int64_t *arg);
  void add_arg(const uint8_t *arg);
  void add_arg(const uint16_t *arg);
  void add_arg(const uint32_t *arg);
  void add_arg(const uint64_t *arg);
  void add_arg(const half_fp *arg);
  void add_arg(const float *arg);
  void add_arg(const double *arg);

  void add_arg(vptr<char> *arg);
  void add_arg(vptr<int8_t> *arg);
  void add_arg(vptr<int16_t> *arg);
  void add_arg(vptr<int32_t> *arg);
  void add_arg(vptr<int64_t> *arg);
  void add_arg(vptr<uint8_t> *arg);
  void add_arg(vptr<uint16_t> *arg);
  void add_arg(vptr<uint32_t> *arg);
  void add_arg(vptr<uint64_t> *arg);
  void add_arg(vptr<half_fp> *arg);
  void add_arg(vptr<float> *arg);
  void add_arg(vptr<double> *arg);
  void add_arg(vptr<void> *arg);

  void add_arg(vptr<const char> *arg);
  void add_arg(vptr<const int8_t> *arg);
  void add_arg(vptr<const int16_t> *arg);
  void add_arg(vptr<const int32_t> *arg);
  void add_arg(vptr<const int64_t> *arg);
  void add_arg(vptr<const uint8_t> *arg);
  void add_arg(vptr<const uint16_t> *arg);
  void add_arg(vptr<const uint32_t> *arg);
  void add_arg(vptr<const uint64_t> *arg);
  void add_arg(vptr<const half_fp> *arg);
  void add_arg(vptr<const float> *arg);
  void add_arg(vptr<const double> *arg);
  void add_arg(vptr<const void> *arg);

 protected:
  explicit DeviceKernel(Device* dev, string name, KernelArgs args);

  virtual void set_arg(uint_tp idx, const bool *arg) = 0;
  virtual void set_arg(uint_tp idx, const char *arg) = 0;
  virtual void set_arg(uint_tp idx, const int8_t *arg) = 0;
  virtual void set_arg(uint_tp idx, const int16_t *arg) = 0;
  virtual void set_arg(uint_tp idx, const int32_t *arg) = 0;
  virtual void set_arg(uint_tp idx, const int64_t *arg) = 0;
  virtual void set_arg(uint_tp idx, const uint8_t *arg) = 0;
  virtual void set_arg(uint_tp idx, const uint16_t *arg) = 0;
  virtual void set_arg(uint_tp idx, const uint32_t *arg) = 0;
  virtual void set_arg(uint_tp idx, const uint64_t *arg) = 0;
  virtual void set_arg(uint_tp idx, const half_fp *arg) = 0;
  virtual void set_arg(uint_tp idx, const float *arg) = 0;
  virtual void set_arg(uint_tp idx, const double *arg) = 0;

  virtual void set_arg(uint_tp idx, vptr<char> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<int8_t> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<int16_t> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<int32_t> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<int64_t> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<uint8_t> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<uint16_t> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<uint32_t> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<uint64_t> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<half_fp> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<float> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<double> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<void> *arg) = 0;

  virtual void set_arg(uint_tp idx, vptr<const char> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<const int8_t> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<const int16_t> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<const int32_t> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<const int64_t> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<const uint8_t> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<const uint16_t> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<const uint32_t> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<const uint64_t> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<const half_fp> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<const float> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<const double> *arg) = 0;
  virtual void set_arg(uint_tp idx, vptr<const void> *arg) = 0;

  Device *device_;
  string name_;
  uint_tp arg_idx_;
  KernelArgs args_;
};

}  // namespace caffe

#endif  // CAFFE_BACKEND_DEVICE_KERNEL_HPP_
