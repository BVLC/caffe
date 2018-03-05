#ifndef CAFFE_QUANTIZER_HPP_
#define CAFFE_QUANTIZER_HPP_

#include <mutex>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/device_program.hpp"

namespace caffe {

struct QuantizerValues {
 public:
  double zero;
  double one;
  double max;
  double min;
  double scale;

  template<typename Dtype>
  typename std::enable_if<float_is_same<Dtype>::value, Dtype>::type
  get_zero() const {
    return static_cast<Dtype>(this->zero);
  }
  template<typename Dtype>
  typename std::enable_if<integer_is_same<Dtype>::value, Dtype>::type
  get_zero() const {
    return static_cast<Dtype>(std::round(this->zero));
  }
  template<typename Dtype>
  typename std::enable_if<float_is_same<Dtype>::value, Dtype>::type
  get_one() const {
    return static_cast<Dtype>(this->one);
  }
  template<typename Dtype>
  typename std::enable_if<integer_is_same<Dtype>::value, Dtype>::type
  get_one() const {
    return static_cast<Dtype>(std::round(this->one));
  }
  template<typename Dtype>
  typename std::enable_if<float_is_same<Dtype>::value, Dtype>::type
  get_max() const {
    return static_cast<Dtype>(this->max);
  }
  template<typename Dtype>
  typename std::enable_if<integer_is_same<Dtype>::value, Dtype>::type
  get_max() const {
    return static_cast<Dtype>(std::round(this->max));
  }
  template<typename Dtype>
  typename std::enable_if<float_is_same<Dtype>::value, Dtype>::type
  get_min() const {
    return static_cast<Dtype>(this->min);
  }
  template<typename Dtype>
  typename std::enable_if<integer_is_same<Dtype>::value, Dtype>::type
  get_min() const {
    return static_cast<Dtype>(std::round(this->min));
  }
  template<typename Dtype>
  typename std::enable_if<float_is_same<Dtype>::value, Dtype>::type
  get_scale() const {
    return static_cast<Dtype>(this->scale);
  }
  template<typename Dtype>
  typename std::enable_if<integer_is_same<Dtype>::value, Dtype>::type
  get_scale() const {
    return static_cast<Dtype>(std::round(this->scale));
  }
};

class QuantizerBase {
 public:
  template<typename Dtype>
  static void MultiplicativeQuantVals(
      const QuantizerValues* lhs, const QuantizerValues* rhs,
      const QuantizerValues* rs, Dtype* rsmult, Dtype* rsshift,
      const uint8_t shift_bits = 0);

  virtual void update() = 0;
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

  virtual void Observe_in(size_t n, const shared_ptr<SyncedMemory> data);
  virtual void Observe_in_cpu(size_t n, const void* data) = 0;
  virtual void Observe_in_gpu(size_t n, vptr<const void> data) = 0;
  virtual void Observe_out(size_t n, const shared_ptr<SyncedMemory> data);
  virtual void Observe_out_cpu(size_t n, const void* data) = 0;
  virtual void Observe_out_gpu(size_t n, vptr<const void> data) = 0;

  virtual bool fw_scale_divide() const = 0;
  virtual bool bw_scale_divide() const = 0;
  virtual bool fw_scale_before_cast() const = 0;
  virtual bool bw_scale_before_cast() const = 0;

  virtual double in_scale_val() = 0;
  virtual double out_scale_val() = 0;

  inline const QuantizerValues& in_quantizer_values() {
    return in_vals_;
  }

  inline const QuantizerValues& out_quantizer_values() {
    return out_vals_;
  }

  inline int_tp index() const {
    return index_;
  }

  inline double observed_max() const {
    return observed_max_;
  }

  inline double observed_min() const {
    return observed_min_;
  }

  inline double in_zero() const {
    return in_vals_.zero;
  }

  inline double in_one() const {
    return in_vals_.one;
  }

  inline double out_zero() const {
    return out_vals_.zero;
  }

  inline double out_one() const {
    return out_vals_.one;
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

  QuantizerValues in_vals_;
  QuantizerValues out_vals_;

  QuantizerMode mode_;
};

template<typename MItype, typename MOtype>
class Quantizer : public QuantizerBase {
 public:
  explicit Quantizer(const QuantizerParameter& param);
  explicit Quantizer(Device* dev_ptr);

  virtual void update();
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

  string fw_gpu_term(int_tp vec_len, string src_var, string tar_var,
                     string scal_var, string in_zero_var,
                     string out_zero_var,
                     string min_in_var, string max_in_var,
                     string min_out_var, string max_out_var) const;
  string bw_gpu_term(int_tp vec_len, string src_var, string tar_var,
                     string scal_var, string in_zero_var,
                     string out_zero_var,
                     string min_in_var, string max_in_var,
                     string min_out_var, string max_out_var) const;

  virtual double in_scale_val();
  virtual double out_scale_val();

 private:
  void init();
};

}  // namespace caffe

#endif  // CAFFE_QUANTIZER_HPP_
