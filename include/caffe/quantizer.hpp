#ifndef CAFFE_QUANTIZER_HPP_
#define CAFFE_QUANTIZER_HPP_

#include <mutex>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/device_program.hpp"

namespace caffe {

class QuantizerBase {
 public:
  virtual void update_param(const QuantizerParameter& param) = 0;

  virtual bool needs_quantization() const = 0;

  virtual void GenerateKernels() = 0;
  virtual void Forward_cpu(size_t n, const void* input,
                           void* output) const = 0;
  virtual void Backward_cpu(size_t n, const void* input,
                            void* output) const = 0;
  virtual void Forward_gpu(size_t n, vptr<const void> input,
                           vptr<void> output) = 0;
  virtual void Backward_gpu(size_t n, vptr<const void> input,
                            vptr<void> output) = 0;

  virtual void Observe_in_cpu(size_t n, const void* data) = 0;
  virtual void Observe_in_gpu(size_t n, vptr<const void> data) = 0;
  virtual void Observe_out_cpu(size_t n, const void* data) = 0;
  virtual void Observe_out_gpu(size_t n, vptr<const void> data) = 0;

  virtual bool fw_scale_divide() const = 0;
  virtual bool bw_scale_divide() const = 0;
  virtual bool fw_scale_before_cast() const = 0;
  virtual bool bw_scale_before_cast() const = 0;

  virtual string fw_scale_term(int_tp vec_len, string scale_var,
                               string src_val) const = 0;
  virtual string bw_scale_term(int_tp vec_len, string scale_var,
                               string src_val) const = 0;

  int_tp get_index() const {
    return index_;
  }

  double get_observed_max() const {
    return observed_max_;
  }

  double get_observed_min() const {
    return observed_min_;
  }

  const QuantizerParameter& quant_param() const {
    return param_;
  }

 protected:
  explicit QuantizerBase(const QuantizerParameter& param);
  explicit QuantizerBase(Device* dev_ptr);
  int_tp index_;

  std::mutex quant_mutex_;

  QuantizerParameter param_;
  shared_ptr<DeviceProgram> program_;
  bool program_ready_;
  Device* device_;

  // Floating point observed range
  double observed_max_;
  double observed_min_;

  // Floating point configured range
  double flt_max_;
  double flt_min_;

  // Input value range
  double max_in_;
  double min_in_;

  // Output value range
  double max_out_;
  double min_out_;

  QuantizerMode mode_;
};

template<typename MItype, typename MOtype>
class Quantizer : public QuantizerBase {
 public:
  explicit Quantizer(const QuantizerParameter& param);
  explicit Quantizer(Device* dev_ptr);

  virtual void update_param(const QuantizerParameter& param);

  virtual bool needs_quantization() const;

  virtual void GenerateKernels();

  void Forward(Blob<MItype>* input, Blob<MOtype>* output,
               bool fw_data, bool fw_diff) const;
  void Backward(Blob<MOtype>* input, Blob<MItype>* output,
                bool bw_data, bool bw_diff) const;
  void Forward_cpu(Blob<MItype>* input, Blob<MOtype>* output,
                   bool fw_data, bool fw_diff) const;
  void Backward_cpu(Blob<MOtype>* input, Blob<MItype>* output,
                    bool bw_data, bool bw_diff) const;
  void Forward_gpu(Blob<MItype>* input, Blob<MOtype>* output,
                   bool fw_data, bool fw_diff);
  void Backward_gpu(Blob<MOtype>* input, Blob<MItype>* output,
                    bool bw_data, bool bw_diff);
  virtual void Forward_cpu(size_t n, const void* input,
                           void* output) const;
  virtual void Backward_cpu(size_t n, const void* input,
                            void* output) const;
  virtual void Forward_gpu(size_t n, vptr<const void> input,
                           vptr<void> output);
  virtual void Backward_gpu(size_t n, vptr<const void> input,
                            vptr<void> output);
  void Forward_cpu(size_t n, const MItype* input,
                   MOtype* output) const;
  void Backward_cpu(size_t n, const MOtype* input,
                    MItype* output) const;
  void Forward_gpu(size_t n, vptr<const MItype> input,
                   vptr<MOtype> output);
  void Backward_gpu(size_t n, vptr<const MOtype> input,
                    vptr<MItype> output);

  virtual void Observe_in_cpu(size_t n, const void* data);
  virtual void Observe_in_gpu(size_t n, vptr<const void> data);
  virtual void Observe_out_cpu(size_t n, const void* data);
  virtual void Observe_out_gpu(size_t n, vptr<const void> data);

  void Observe_in_cpu(size_t n, const MItype* data);
  void Observe_in_gpu(size_t n, vptr<const MItype> data);
  void Observe_out_cpu(size_t n, const MOtype* data);
  void Observe_out_gpu(size_t n, vptr<const MOtype> data);

  virtual bool fw_scale_divide() const;
  virtual bool bw_scale_divide() const;
  virtual bool fw_scale_before_cast() const;
  virtual bool bw_scale_before_cast() const;

  MItype fw_scale_before_cast_val() const;
  MOtype fw_scale_after_cast_val() const;
  MOtype bw_scale_before_cast_val() const;
  MItype bw_scale_after_cast_val() const;

  virtual string fw_scale_term(int_tp vec_len, string scale_var,
                               string src_val) const;
  virtual string bw_scale_term(int_tp vec_len, string scale_var,
                               string src_val) const;
 private:
  void init();
};

}  // namespace caffe

#endif  // CAFFE_QUANTIZER_HPP_
