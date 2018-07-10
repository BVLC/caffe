#ifndef CAFFE_QUANTIZER_HPP_
#define CAFFE_QUANTIZER_HPP_

#include <mutex>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/device_program.hpp"
#include "caffe/util/type_utils.hpp"

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

  template<typename Dtype>
  void auto_min_max() {
    this->min = type_min_val<Dtype>();
    this->max = type_max_val<Dtype>();
  }


  template<typename Dtype>
  float scale_val(float flt_min, float flt_max) {
    if (is_float_type<Dtype>()) {
      return 1.0;
    } else {
      return (flt_max - flt_min) / (this->max - this->min);
    }
  }

  template<typename Dtype>
  void compute_values(float flt_min, float flt_max) {
    if (is_float_type<Dtype>()) {
      this->zero = 0.0;
      this->one = 1.0;
    } else {
      double initial_zero = this->min - flt_min
          / this->scale_val<Dtype>(flt_min, flt_max);
      if (initial_zero < this->min) {
        this->zero = this->min;
      } else if (initial_zero > this->max) {
        this->zero = this->max;
      } else {
        this->zero = std::round(initial_zero);
      }
      this->one = 1.0/scale_val<Dtype>(flt_min, flt_max) + this->zero;
    }
    this->scale = scale_val<Dtype>(flt_min, flt_max);
  }

  // For testing purposes
  void print() const {
    std::cout << "Zero: " << this->zero << std::endl;
    std::cout << "One: " << this->one << std::endl;
    std::cout << "Min: " << this->min << std::endl;
    std::cout << "Max: " << this->max << std::endl;
    std::cout << "Scale: " << this->scale << std::endl;
  }
};

class QuantizerBase {
 public:
  template<typename Dtype>
  static void ScaleQuantVals(
      const QuantizerValues* const lhs, const QuantizerValues* const rhs,
      Dtype* rsmult, int8_t* rsshift, const uint8_t shift_bits = 31);
  template<typename Dtype>
  static void MultiplicativeQuantVals(
      const QuantizerValues* const lhs, const QuantizerValues* const rhs,
      const QuantizerValues* const rs, Dtype* rsmult, int8_t* rsshift,
      const uint8_t shift_bits = 31);

  virtual void update() = 0;
  virtual void update_param(const QuantizerParameter& param) = 0;
  void reset_observed_values();

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

  virtual void PseudoQuantIn(size_t n, const shared_ptr<SyncedMemory> input,
                             shared_ptr<SyncedMemory> output);
  virtual void PseudoQuantIn_cpu(size_t n, const void* input, void* output) = 0;
  virtual void PseudoQuantIn_gpu(size_t n, vptr<const void> input,
                                 vptr<void> output) = 0;
  virtual void PseudoQuantOut(size_t n, const shared_ptr<SyncedMemory> input,
                              shared_ptr<SyncedMemory> output);
  virtual void PseudoQuantOut_cpu(size_t n, const void* input,
                                  void* output) = 0;
  virtual void PseudoQuantOut_gpu(size_t n, vptr<const void> input,
                                  vptr<void> output) = 0;

  virtual void ObserveIn(size_t n, const shared_ptr<SyncedMemory> data,
                         bool force = false);
  virtual void ObserveIn_cpu(size_t n, const void* data,
                             bool force = false) = 0;
  virtual void ObserveIn_gpu(size_t n, vptr<const void> data,
                             bool force = false) = 0;
  virtual void ObserveOut(size_t n, const shared_ptr<SyncedMemory> data,
                          bool force = false);
  virtual void ObserveOut_cpu(size_t n, const void* data,
                              bool force = false) = 0;
  virtual void ObserveOut_gpu(size_t n, vptr<const void> data,
                              bool force = false) = 0;

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


  virtual void PseudoQuantIn_cpu(size_t n, const void* input, void* output);
  virtual void PseudoQuantIn_gpu(size_t n, vptr<const void> input,
                                 vptr<void> output);
  virtual void PseudoQuantOut_cpu(size_t n, const void* input,
                                  void* output);
  virtual void PseudoQuantOut_gpu(size_t n, vptr<const void> input,
                                  vptr<void> output);

  virtual void PseudoQuantIn_cpu(size_t n, const MItype* input, MItype* output);
  virtual void PseudoQuantIn_gpu(size_t n, vptr<const MItype> input,
                                 vptr<MItype> output);
  virtual void PseudoQuantOut_cpu(size_t n, const MOtype* input,
                                  MOtype* output);
  virtual void PseudoQuantOut_gpu(size_t n, vptr<const MOtype> input,
                                  vptr<MOtype> output);

  virtual void ObserveIn_cpu(size_t n, const void* data, bool force = false);
  virtual void ObserveIn_gpu(size_t n, vptr<const void> data,
                             bool force = false);
  virtual void ObserveOut_cpu(size_t n, const void* data, bool force = false);
  virtual void ObserveOut_gpu(size_t n, vptr<const void> data,
                              bool force = false);

  void ObserveIn_cpu(size_t n, const MItype* data, bool force = false);
  void ObserveIn_gpu(size_t n, vptr<const MItype> data, bool force = false);
  void ObserveOut_cpu(size_t n, const MOtype* data, bool force = false);
  void ObserveOut_gpu(size_t n, vptr<const MOtype> data, bool force = false);

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
                     string min_out_var, string max_out_var) const;
  string bw_gpu_term(int_tp vec_len, string src_var, string tar_var,
                     string scal_var, string in_zero_var,
                     string out_zero_var,
                     string min_out_var, string max_out_var) const;

  virtual double in_scale_val();
  virtual double out_scale_val();

 private:
  void init();
};

}  // namespace caffe

#endif  // CAFFE_QUANTIZER_HPP_
