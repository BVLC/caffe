#ifndef CAFFE_QUANTIZER_HPP_
#define CAFFE_QUANTIZER_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/device_program.hpp"

namespace caffe {

enum QuantizerMode {
  // In passive mode, the quantizer does nothing
  QUANTIZER_MODE_PASSIVE = 0,
  // In observe mode, the quantizer observes max/min values
  QUANTIZER_MODE_OBSERVE = 1,
  // In active mode, the quantizer actively converts from one type to another
  QUANTIZER_MODE_ACTIVE = 2,
  // Active and observing
  QUANTIZER_MODE_ACTIVE_OBSERVE = 3
};

class QuantizerBase {
 public:
  QuantizerMode get_mode() const;
  void set_mode(QuantizerMode mode);
  string get_mode_string();

  virtual void GenerateKernels() = 0;
  virtual void Forward_cpu(size_t n, const void* input, void* output) = 0;
  virtual void Backward_cpu(size_t n, const void* input, void* output) = 0;
  virtual void Forward_gpu(size_t n, vptr<const void> input,
                           vptr<void> output) = 0;
  virtual void Backward_gpu(size_t n, vptr<const void> input,
                            vptr<void> output) = 0;

  virtual bool fw_scale_divide() const = 0;
  virtual bool bw_scale_divide() const = 0;
  virtual bool fw_scale_before_cast() const = 0;
  virtual bool bw_scale_before_cast() const = 0;

  virtual string fw_scale_term(int_tp vec_len, string scale_var,
                               string src_val) const = 0;
  virtual string bw_scale_term(int_tp vec_len, string scale_var,
                               string src_val) const = 0;

 protected:
  explicit QuantizerBase(QuantizerParameter& param);
  QuantizerParameter quant_param_;
  shared_ptr<DeviceProgram> program_;
  bool program_ready_;
  Device* device_;

  double observed_max_;
  double observed_min_;

  double max_in_;
  double min_in_;
  double max_out_;
  double min_out_;

  QuantizerMode mode_;
};

template<typename MItype, typename MOtype>
class Quantizer : public QuantizerBase {
 public:
  explicit Quantizer(QuantizerParameter& param);

  virtual void GenerateKernels();

  void Forward(Blob<MItype>* input, Blob<MOtype>* output,
               bool fw_data, bool fw_diff);
  void Backward(Blob<MOtype>* input, Blob<MItype>* output,
                bool bw_data, bool bw_diff);
  void Forward_cpu(Blob<MItype>* input, Blob<MOtype>* output,
                   bool fw_data, bool fw_diff);
  void Backward_cpu(Blob<MOtype>* input, Blob<MItype>* output,
                    bool bw_data, bool bw_diff);
  void Forward_gpu(Blob<MItype>* input, Blob<MOtype>* output,
                   bool fw_data, bool fw_diff);
  void Backward_gpu(Blob<MOtype>* input, Blob<MItype>* output,
                    bool bw_data, bool bw_diff);
  virtual void Forward_cpu(size_t n, const void* input, void* output);
  virtual void Backward_cpu(size_t n, const void* input, void* output);
  virtual void Forward_gpu(size_t n, vptr<const void> input,
                           vptr<void> output);
  virtual void Backward_gpu(size_t n, vptr<const void> input,
                            vptr<void> output);
  void Forward_cpu(size_t n, const MItype* input, MOtype* output);
  void Backward_cpu(size_t n, const MOtype* input, MItype* output);
  void Forward_gpu(size_t n, vptr<const MItype> input, vptr<MOtype> output);
  void Backward_gpu(size_t n, vptr<const MOtype> input, vptr<MItype> output);

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
};

}  // namespace caffe

#endif  // CAFFE_QUANTIZER_HPP_
