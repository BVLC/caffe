#include <cmath>
#include <algorithm>

#include "caffe/backend/device.hpp"
#include "caffe/quantizer.hpp"
#include "caffe/util/type_utils.hpp"

namespace caffe {

QuantizerBase::QuantizerBase(const QuantizerParameter& param)
  : param_(param), program_ready_(false) {
  this->device_ = Caffe::GetDevice(param_.device(), true);
}

QuantizerBase::QuantizerBase(Device* dev_ptr)
  : param_(), program_ready_(false) {
  this->device_ = dev_ptr;
}

template<typename Dtype>
void QuantizerBase::MultiplicativeQuantVals(
    const QuantizerValues* const lhs, const QuantizerValues* const rhs,
    const QuantizerValues* const rs, Dtype* rsmult, int8_t* rsshift,
    const uint8_t shift_bits) {
  double multiplier = ((lhs ? lhs->scale : 1.0)  *
                       (rhs ? rhs->scale : 1.0)) /
                       (rs ? rs->scale : 1.0);
  if (multiplier == 0.0) {
    *rsmult = Dtype(0);
    *rsshift = int8_t(0);
  }
  if (is_float_type<Dtype>()) {
    // Float can be scaled without shift
    *rsmult = static_cast<Dtype>(multiplier);
  } else {
    int64_t lshift = shift_bits ? (static_cast<int64_t>(shift_bits)) :
                                  (sizeof(Dtype) * 8ll - 1ll);
    Dtype s = 0;
    while (multiplier < 0.5 && s < lshift) {
      multiplier *= 2.0;
      ++s;
    }
    while (multiplier > 1.0 && s > -lshift) {
      multiplier /= 2.0;
      --s;
    }
    int64_t q = static_cast<int64_t>(std::round(multiplier
                                                * double(1ll << lshift)));
    if (q == (1ll << lshift)) {
      q /= 2;
      --s;
    }
    CHECK_LE(q, type_max_integer_representable<Dtype>());
    *rsmult = static_cast<Dtype>(q);
    *rsshift = s;
  }
}

INSTANTIATE_FUNC_1T(QuantizerBase::MultiplicativeQuantVals,
                            (int8_t)(int16_t)(int32_t)(int64_t));
INSTANTIATE_FUNC_1T(QuantizerBase::MultiplicativeQuantVals,
                            (half_fp)(float)(double));

void QuantizerBase::Observe_in(size_t n, const shared_ptr<SyncedMemory> data) {
  if (mode_ == CAFFE_QUANT_PASSIVE) {
    return;
  }
  switch(data->head()) {
    case SyncedMemory::HEAD_AT_CPU:
      this->Observe_in_cpu(n, data->cpu_data());
      break;
    case SyncedMemory::HEAD_AT_GPU:
    case SyncedMemory::SYNCED:
    case SyncedMemory::UNINITIALIZED:
      if (this->device_->backend() == BACKEND_CPU) {
        this->Observe_in_cpu(n, data->cpu_data());
      } else {
        this->Observe_in_gpu(n, data->gpu_data());
      }
      break;
    default:
      LOG(FATAL) << "SyncedMemory in invalid state";
  }
}

void QuantizerBase::Observe_out(size_t n, const shared_ptr<SyncedMemory> data) {
  if (mode_ == CAFFE_QUANT_PASSIVE) {
    return;
  }
  switch(data->head()) {
    case SyncedMemory::HEAD_AT_CPU:
      this->Observe_out_cpu(n, data->cpu_data());
      break;
    case SyncedMemory::HEAD_AT_GPU:
    case SyncedMemory::SYNCED:
    case SyncedMemory::UNINITIALIZED:
      if (this->device_->backend() == BACKEND_CPU) {
        this->Observe_out_cpu(n, data->cpu_data());
      } else {
        this->Observe_out_gpu(n, data->gpu_data());
      }
      break;
    default:
      LOG(FATAL) << "SyncedMemory in invalid state";
  }
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::update() {
  this->param_.set_observed_max(this->observed_max_);
  this->param_.set_observed_min(this->observed_min_);
  this->update_param(this->param_);
}


template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::update_param(const QuantizerParameter& param) {
  quant_mutex_.lock();
  this->param_ = param;
  this->param_.set_input_data_type(proto_data_type<MItype>());
  this->param_.set_output_data_type(proto_data_type<MOtype>());
  this->program_ready_ = false;
  this->mode_ = param_.mode();

  // Load estimated max/min values
  if (this->param_.has_observed_min()) {
    this->observed_min_ = param_.observed_min();
  }
  if (this->param_.has_observed_max()) {
    this->observed_max_ = param_.observed_max();
  }

  // Set floating point maximum and minimum
  this->flt_min_ = this->observed_min_;
  this->flt_max_ = this->observed_max_;

  if (this->param_.has_min_in()) {
    this->in_vals_.min = param_.min_in();
  } else {
    if (is_float_type<MItype>()) {
      this->in_vals_.min = this->flt_min_;
    }
    if (is_unsigned_integer_type<MItype>()) {
      this->in_vals_.min = type_min_val<MItype>();
    }
  }
  if (this->param_.has_max_in()) {
    this->in_vals_.max = param_.max_in();
  } else {
    if (is_float_type<MItype>()) {
      this->in_vals_.max = this->flt_max_;
    }
    if (is_unsigned_integer_type<MItype>()) {
      this->in_vals_.max = type_max_val<MItype>();
    }
  }

  if (this->param_.has_min_out()) {
    this->out_vals_.min = param_.min_out();
  } else {
    if (is_float_type<MOtype>()) {
      this->out_vals_.min = this->flt_min_;
    }
    if (is_unsigned_integer_type<MOtype>()) {
      this->out_vals_.min = type_min_val<MOtype>();
    }
  }
  if (this->param_.has_max_out()) {
    this->out_vals_.max = param_.max_out();
  } else {
    if (is_float_type<MOtype>()) {
      this->out_vals_.max = this->flt_max_;
    }
    if (is_unsigned_integer_type<MOtype>()) {
      this->out_vals_.max = type_max_val<MOtype>();
    }
  }

  if (is_float_type<MItype>()) {
    this->in_vals_.zero = 0.0;
    this->in_vals_.one = 1.0;
  } else {
    double initial_zero = this->in_vals_.min - this->flt_min_
        / this->in_scale_val();
    if (initial_zero < this->in_vals_.min) {
      this->in_vals_.zero = this->in_vals_.min;
    } else if (initial_zero > this->in_vals_.max) {
      this->in_vals_.zero = this->in_vals_.max;
    } else {
      this->in_vals_.zero = std::round(initial_zero);
    }
    this->in_vals_.one = 1.0/in_scale_val() + this->in_vals_.zero;
  }
  this->in_vals_.scale = in_scale_val();

  if (is_float_type<MOtype>()) {
    this->out_vals_.zero = 0.0;
    this->out_vals_.one = 1.0;
  } else {
    double initial_zero = this->out_vals_.min - this->flt_min_
        / this->out_scale_val();
    if (initial_zero < this->out_vals_.min) {
      this->out_vals_.zero = this->out_vals_.min;
    } else if (initial_zero > this->out_vals_.max) {
      this->out_vals_.zero = this->out_vals_.max;
    } else {
      this->out_vals_.zero = std::round(initial_zero);
    }
    this->out_vals_.one = 1.0/out_scale_val() + this->out_vals_.zero;
  }
  this->out_vals_.scale = out_scale_val();

  // Need to recompile after changes to the quantizer parameters
  program_ready_ = false;
  quant_mutex_.unlock();
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::init() {
  this->program_ = this->device_->CreateProgram();

  this->mode_ = CAFFE_QUANT_PASSIVE;

  this->observed_min_ = 0.0;
  this->observed_max_ = 0.0;

  this->flt_min_ = 0.0;
  this->flt_max_ = 0.0;
  this->in_vals_.min = MItype(0);
  this->in_vals_.max = MItype(0);
  this->out_vals_.min = MOtype(0);
  this->out_vals_.max = MOtype(0);
  this->in_vals_.zero = 0.0;
  this->out_vals_.zero = 0.0;
}

template<typename MItype, typename MOtype>
bool Quantizer<MItype, MOtype>::needs_quantization() const {
  if (is_float_type<MItype>() && is_float_type<MOtype>()) {
    // Float types can be cast to each other
    return false;
  }
  if ((is_integer_type<MItype>() && is_float_type<MOtype>())
       || (is_integer_type<MOtype>() && is_float_type<MItype>())) {
    // Float to integer and inverse needs quantizations
    return true;
  }
  if (is_integer_type<MItype>() && is_integer_type<MOtype>()) {
    if (std::is_same<MItype, MOtype>::value) {
      if (this->in_vals_.min == this->out_vals_.min &&
          this->in_vals_.max == this->out_vals_.max &&
          this->in_vals_.zero == this->out_vals_.zero) {
        // Identical integer type with same value range does not need quantization
        return false;
      }
    }
    // Integer types of different sizes need rescaling
    return true;
  }
  return true;
}

template<typename MItype, typename MOtype>
double Quantizer<MItype, MOtype>::in_scale_val() {
  if (is_float_type<MItype>()) {
    return 1.0;
  } else {
    return (this->flt_max_ - this->flt_min_)
        / (this->in_vals_.max - this->in_vals_.min);
  }
}

template<typename MItype, typename MOtype>
double Quantizer<MItype, MOtype>::out_scale_val() {
  if (is_float_type<MOtype>()) {
    return 1.0;
  } else {
    return (this->flt_max_ - this->flt_min_)
        / (this->out_vals_.max - this->out_vals_.min);
  }
}


template<typename MItype, typename MOtype>
Quantizer<MItype, MOtype>::Quantizer(const QuantizerParameter& param)
  : QuantizerBase(param) {
  init();
  update_param(param);
}

template<typename MItype, typename MOtype>
Quantizer<MItype, MOtype>::Quantizer(Device* dev_ptr)
  : QuantizerBase(dev_ptr) {
  init();
  QuantizerParameter param;
  param.set_device(dev_ptr->id());
  param.set_input_data_type(proto_data_type<MItype>());
  param.set_output_data_type(proto_data_type<MOtype>());
  param.set_zone(0);
  update_param(param);
}


template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Forward_cpu(Blob<MItype>* input,
                                            Blob<MOtype>* output,
                                            bool fw_data, bool fw_diff) const {
  CHECK_EQ(input->count(), output->count());
  if (fw_data) {
    this->Forward_cpu(input->count(),
                      input->cpu_data(),
                      output->mutable_cpu_data());
  }
  if (fw_diff) {
    this->Forward_cpu(input->count(),
                      input->cpu_diff(),
                      output->mutable_cpu_diff());
  }
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Forward_cpu(
   size_t n, const void* input, void* output) const {
  this->Forward_cpu(n,
                    static_cast<const MItype*>(input),
                    static_cast<MOtype*>(output));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Forward_cpu(
   size_t n, const MItype* input, MOtype* output) const {
  CHECK(input);
  CHECK(output);

  if (!needs_quantization()) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      output[i] = static_cast<MOtype>(input[i]);
    }
    return;
  }

  if (fw_scale_before_cast()) {
    typedef typename std::conditional<float_is_same<MItype>::value, MItype,
            typename std::conditional<sizeof(MItype) == 1, int16_t,
            typename std::conditional<sizeof(MItype) == 2, int32_t, int64_t>
                                                  ::type>::type>::type Difftype;

    const Difftype scal = static_cast<Difftype>(fw_scale_before_cast_val());
    const Difftype in_zero = static_cast<Difftype>(this->in_vals_.zero);
    const Difftype out_zero = static_cast<Difftype>(this->out_vals_.zero);
    const Difftype min_out = static_cast<Difftype>(this->out_vals_.min);
    const Difftype max_out = static_cast<Difftype>(this->out_vals_.max);
    if (fw_scale_divide()) {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        Difftype centered_input = static_cast<Difftype>(input[i]) - in_zero;
        output[i] = static_cast<MOtype>(std::min(std::max(
            static_cast<Difftype>(centered_input / scal + out_zero),
            min_out), max_out));
      }
    } else {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        Difftype centered_input = static_cast<Difftype>(input[i]) - in_zero;
        output[i] = static_cast<MOtype>(std::min(std::max(
            static_cast<Difftype>(centered_input * scal + out_zero),
            min_out), max_out));
      }
    }
  } else {
    typedef typename std::conditional<float_is_same<MOtype>::value, MOtype,
            typename std::conditional<sizeof(MOtype) == 1, int16_t,
            typename std::conditional<sizeof(MOtype) == 2, int32_t, int64_t>
                                                  ::type>::type>::type Difftype;

    const Difftype scal = static_cast<Difftype>(fw_scale_after_cast_val());
    const Difftype in_zero = static_cast<Difftype>(this->in_vals_.zero);
    const Difftype out_zero = static_cast<Difftype>(this->out_vals_.zero);
    const Difftype min_out = static_cast<Difftype>(this->out_vals_.min);
    const Difftype max_out = static_cast<Difftype>(this->out_vals_.max);
    if (fw_scale_divide()) {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        Difftype centered_output = (static_cast<Difftype>(input[i]) - in_zero)
                                   / scal;
        bool uf = (centered_output < min_out - out_zero) && out_zero < 0;
        bool of = (centered_output > max_out - out_zero) && out_zero > 0;
        output[i] = static_cast<MOtype>(uf ? min_out :
                                       (of ? max_out :
                                        centered_output + out_zero));
      }
    } else {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        Difftype centered_output = (static_cast<Difftype>(input[i]) - in_zero)
                                   * scal;
        bool uf = (centered_output < min_out - out_zero) && out_zero < 0;
        bool of = (centered_output > max_out - out_zero) && out_zero > 0;
        output[i] = static_cast<MOtype>(uf ? min_out :
                                       (of ? max_out :
                                        centered_output + out_zero));
      }
    }
  }
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Backward_cpu(Blob<MOtype>* input,
                                             Blob<MItype>* output,
                                             bool bw_data, bool bw_diff) const {
  CHECK_EQ(input->count(), output->count());
  if (bw_data) {
    this->Backward_cpu(input->count(),
                       input->cpu_data(),
                       output->mutable_cpu_data());
  }
  if (bw_diff) {
    this->Backward_cpu(input->count(),
                       input->cpu_diff(),
                       output->mutable_cpu_diff());
  }
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Backward_cpu(
   size_t n, const void* input, void* output) const {
  this->Backward_cpu(n,
                     static_cast<const MOtype*>(input),
                     static_cast<MItype*>(output));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Backward_cpu(
   size_t n, const MOtype* input, MItype* output) const {
  CHECK(input);
  CHECK(output);

  if (!needs_quantization()) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      output[i] = static_cast<MItype>(input[i]);
    }
    return;
  }

  if (bw_scale_before_cast()) {
    typedef typename std::conditional<float_is_same<MOtype>::value, MOtype,
            typename std::conditional<sizeof(MOtype) == 1, int16_t,
            typename std::conditional<sizeof(MOtype) == 2, int32_t, int64_t>
                                                  ::type>::type>::type Difftype;

    const Difftype scal = static_cast<Difftype>(bw_scale_before_cast_val());
    const Difftype in_zero = static_cast<Difftype>(this->out_vals_.zero);
    const Difftype out_zero = static_cast<Difftype>(this->in_vals_.zero);
    const Difftype min_out = static_cast<Difftype>(this->in_vals_.min);
    const Difftype max_out = static_cast<Difftype>(this->in_vals_.max);
    if (bw_scale_divide()) {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        Difftype centered_input = static_cast<Difftype>(input[i]) - in_zero;
        output[i] = static_cast<MItype>(std::min(std::max(
            static_cast<Difftype>(centered_input / scal + out_zero),
            min_out), max_out));
      }
    } else {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        Difftype centered_input = static_cast<Difftype>(input[i]) - in_zero;
        output[i] = static_cast<MItype>(std::min(std::max(
            static_cast<Difftype>(centered_input * scal + out_zero),
            min_out), max_out));
      }
    }
  } else {
    typedef typename std::conditional<float_is_same<MItype>::value, MItype,
            typename std::conditional<sizeof(MItype) == 1, int16_t,
            typename std::conditional<sizeof(MItype) == 2, int32_t, int64_t>
                                                  ::type>::type>::type Difftype;

    const Difftype scal = static_cast<Difftype>(bw_scale_after_cast_val());
    const Difftype in_zero = static_cast<Difftype>(this->out_vals_.zero);
    const Difftype out_zero = static_cast<Difftype>(this->in_vals_.zero);
    const Difftype min_out = static_cast<Difftype>(this->in_vals_.min);
    const Difftype max_out = static_cast<Difftype>(this->in_vals_.max);
    if (bw_scale_divide()) {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        Difftype centered_output = (static_cast<Difftype>(input[i]) - in_zero)
                                   / scal;
        bool uf = (centered_output < min_out - out_zero) && out_zero < 0;
        bool of = (centered_output > max_out - out_zero) && out_zero > 0;
        output[i] = static_cast<MItype>(uf ? min_out :
                                       (of ? max_out :
                                        centered_output + out_zero));
      }
    } else {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        Difftype centered_output = (static_cast<Difftype>(input[i]) - in_zero)
                                   * scal;
        bool uf = (centered_output < min_out - out_zero) && out_zero < 0;
        bool of = (centered_output > max_out - out_zero) && out_zero > 0;
        output[i] = static_cast<MItype>(uf ? min_out :
                                       (of ? max_out :
                                        centered_output + out_zero));
      }
    }
  }
}

// Use division instead of multiplication for scaling
template<typename MItype, typename MOtype>
bool Quantizer<MItype, MOtype>::fw_scale_divide() const {
  if ((is_integer_type<MItype>() && is_integer_type<MOtype>())
      || (is_float_type<MItype>() && is_float_type<MOtype>())) {
    if (safe_sizeof<MItype>() > safe_sizeof<MOtype>()) {
      // Scale to smaller type with division
      return true;
    } else {
      // Scale to larger type with multiplication
      return false;
    }
  }
  if (is_integer_type<MItype>() && is_float_type<MOtype>()) {
    // Use division in float type (after casting)
    return true;
  }
  if (is_float_type<MItype>() && is_integer_type<MOtype>()) {
    // Use multiplication in float type (before casting)
    return false;
  }
  return false;
}

template<typename MItype, typename MOtype>
bool Quantizer<MItype, MOtype>::bw_scale_divide() const {
  if ((is_integer_type<MItype>() && is_integer_type<MOtype>())
      || (is_float_type<MItype>() && is_float_type<MOtype>())) {
    if (safe_sizeof<MItype>() > safe_sizeof<MOtype>()) {
      // Scale from smaller type with multiplication
      return false;
    } else {
      // Scale from larger type with division
      return true;
    }
  }
  if (is_integer_type<MItype>() && is_float_type<MOtype>()) {
    // Use multiplication in float type (before casting)
    return false;
  }
  if (is_float_type<MItype>() && is_integer_type<MOtype>()) {
    // Use division in float type (after casting)
    return true;
  }
  return false;
}

// Scale before or after casting; avoid overflow / clipping
template<typename MItype, typename MOtype>
bool Quantizer<MItype, MOtype>::fw_scale_before_cast() const {
  if ((is_integer_type<MItype>() && is_integer_type<MOtype>())
      || (is_float_type<MItype>() && is_float_type<MOtype>())) {
    if (safe_sizeof<MItype>() > safe_sizeof<MOtype>()) {
      return true;
    } else {
      return false;
    }
  }
  if (is_integer_type<MItype>() && is_float_type<MOtype>()) {
    // Cast to float, then scale in float
    return false;
  }
  if (is_float_type<MItype>() && is_integer_type<MOtype>()) {
    // Scale in float, then cast to integer
    return true;
  }
  return true;
}

template<typename MItype, typename MOtype>
bool Quantizer<MItype, MOtype>::bw_scale_before_cast() const {
  if ((is_integer_type<MItype>() && is_integer_type<MOtype>())
      || (is_float_type<MItype>() && is_float_type<MOtype>())) {
    if (safe_sizeof<MItype>() > safe_sizeof<MOtype>()) {
      return false;
    } else {
      return true;
    }
  }
  if (is_integer_type<MItype>() && is_float_type<MOtype>()) {
    // Scale in float, then cast to integer
    return true;
  }
  if (is_float_type<MItype>() && is_integer_type<MOtype>()) {
    // Cast to float, then scale in float
    return false;
  }
  return true;
}

template<typename MItype, typename MOtype>
MItype Quantizer<MItype, MOtype>::fw_scale_before_cast_val() const {
  double val = 0.0;
  if (this->fw_scale_divide()) {
    val = (this->in_vals_.max - this->in_vals_.min)
      / (this->out_vals_.max - this->out_vals_.min);
  } else {
    val = (this->out_vals_.max - this->out_vals_.min)
      / (this->in_vals_.max - this->in_vals_.min);
  }
  return static_cast<MItype>(val);
}

template<typename MItype, typename MOtype>
MOtype Quantizer<MItype, MOtype>::fw_scale_after_cast_val() const {
  double val = 0.0;
  if (this->fw_scale_divide()) {
    val = (this->in_vals_.max - this->in_vals_.min)
      / (this->out_vals_.max - this->out_vals_.min);
  } else {
    val = (this->out_vals_.max - this->out_vals_.min)
      / (this->in_vals_.max - this->in_vals_.min);
  }
  return static_cast<MOtype>(val);
}

template<typename MItype, typename MOtype>
MOtype Quantizer<MItype, MOtype>::bw_scale_before_cast_val() const {
  double val = 0.0;
  if (this->bw_scale_divide()) {
    val = (this->out_vals_.max - this->out_vals_.min)
      / (this->in_vals_.max - this->in_vals_.min);
  } else {
    val = (this->in_vals_.max - this->in_vals_.min)
      / (this->out_vals_.max - this->out_vals_.min);
  }
  return static_cast<MOtype>(val);
}

template<typename MItype, typename MOtype>
MItype Quantizer<MItype, MOtype>::bw_scale_after_cast_val() const {
  double val = 0.0;
  if (this->bw_scale_divide()) {
    val = (this->out_vals_.max - this->out_vals_.min)
      / (this->in_vals_.max - this->in_vals_.min);
  } else {
    val = (this->in_vals_.max - this->in_vals_.min)
      / (this->out_vals_.max - this->out_vals_.min);
  }
  return static_cast<MItype>(val);
}

template<typename MItype, typename MOtype>
string quant_gpu_term(DeviceProgram* program, bool needs_quantization,
                      bool scale_divide, bool scale_before_cast,
                      int_tp vec_len, string src_var, string tar_var,
                      string scal_var, string in_zero_var, string out_zero_var,
                      string min_out_var, string max_out_var) {
  stringstream ss;
   string op = "";
   if (!needs_quantization) {
     if (std::is_same<MItype, MOtype>::value) {
       ss << tar_var << " = " << src_var << ";" << std::endl;
       return ss.str();
     } else {
       ss << tar_var << " = "
          << program->template convert_type<MOtype>(vec_len, src_var)
          << ";" << std::endl;
       return ss.str();
     }
   } else {
     if (scale_divide) {
       op = " / ";
     } else {
       op = " * ";
     }
   }

   stringstream ss_s0;
   ss << "{" << std::endl;
   if (scale_before_cast) {
     typedef typename std::conditional<float_is_same<MItype>::value, MItype,
             typename std::conditional<sizeof(MItype) == 1, int16_t,
             typename std::conditional<sizeof(MItype) == 2, int32_t, int64_t>
                                                 ::type>::type>::type Difftype;

     string min_gpu_func = "min";
     string max_gpu_func = "max";
     if (program->device()->backend() == BACKEND_OPENCL &&
         is_float_type<Difftype>()) {
       min_gpu_func = "fmin";
       max_gpu_func = "fmax";
     }

     ss << (program->template device_type_name<Difftype>())
        <<  (vec_len > 0 ? std::to_string(vec_len) : "")
        << " gpu_centered_input = "
        << program->template convert_type<Difftype>(vec_len, src_var) << " - "
        << in_zero_var << ";" << std::endl;

     ss_s0 << min_gpu_func << "(" << max_gpu_func << "(gpu_centered_input"
           << op << scal_var << " + "  << out_zero_var << "," << min_out_var
           << "), " << max_out_var << ")";

     ss << tar_var << " = " << program->template convert_type<MOtype>(vec_len,
                               ss_s0.str()) << ";" << std::endl;
   } else {
     typedef typename std::conditional<float_is_same<MOtype>::value, MOtype,
             typename std::conditional<sizeof(MOtype) == 1, int16_t,
             typename std::conditional<sizeof(MOtype) == 2, int32_t, int64_t>
                                                 ::type>::type>::type Difftype;

     string min_gpu_func = "min";
     string max_gpu_func = "max";
     if (program->device()->backend() == BACKEND_OPENCL &&
         is_float_type<Difftype>()) {
       min_gpu_func = "fmin";
       max_gpu_func = "fmax";
     }

     ss << (program->template device_type_name<Difftype>())
        <<  (vec_len > 0 ? std::to_string(vec_len) : "")
        << " gpu_centered_output = ("
        << program->template convert_type<Difftype>(vec_len, src_var) << " - "
        << in_zero_var << ")" << op << scal_var << ";" << std::endl;

     ss << "int" << (vec_len > 0 ? std::to_string(vec_len) : "")
        << " gpu_uf = (gpu_centered_output < "
        << min_out_var << " - " << out_zero_var << ") && " << out_zero_var
        << " < 0;" << std::endl;
     ss << "int" << (vec_len > 0 ? std::to_string(vec_len) : "")
        << " gpu_of = (" << src_var << " > "
        << max_out_var << " - " << out_zero_var << ") && " << out_zero_var
        << " > 0;" << std::endl;
     ss_s0 << "gpu_uf ? " << min_out_var << " : (gpu_of ? "
        << max_out_var << " : gpu_centered_output + " << out_zero_var << ")";
     ss << tar_var << " = "
        << program->template convert_type<MOtype>(vec_len, ss_s0.str()) << ";"
        << std::endl;
   }
   ss << "}" << std::endl;
   return ss.str();
}

template<typename MItype, typename MOtype>
string Quantizer<MItype, MOtype>::fw_gpu_term(
    int_tp vec_len, string src_var, string tar_var,
    string scal_var, string in_zero_var, string out_zero_var,
    string min_out_var, string max_out_var) const {
  return quant_gpu_term<MItype, MOtype>(this->program_.get(),
              needs_quantization(), fw_scale_divide(),
              fw_scale_before_cast(), vec_len, src_var, tar_var, scal_var,
              in_zero_var, out_zero_var, min_out_var, max_out_var);
}

template<typename MItype, typename MOtype>
string Quantizer<MItype, MOtype>::bw_gpu_term(
    int_tp vec_len, string src_var, string tar_var,
    string scal_var, string in_zero_var, string out_zero_var,
    string min_out_var, string max_out_var) const {
  return quant_gpu_term<MOtype, MItype>(this->program_.get(),
              needs_quantization(), bw_scale_divide(),
              bw_scale_before_cast(), vec_len, src_var, tar_var, scal_var,
              in_zero_var, out_zero_var, min_out_var, max_out_var);
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Observe_in_cpu(size_t n, const void* data) {
  Observe_in_cpu(n, static_cast<const MItype*>(data));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Observe_in_cpu(size_t n, const MItype* data) {
  if (mode_ == CAFFE_QUANT_PASSIVE) {
    return;
  }
  double local_min = type_max_val<double>();
  double local_max = type_min_val<double>();
  double scal = in_scale_val();
  double zero = in_zero();
  for (size_t i = 0; i < n; ++i) {
    double value = (static_cast<double>(data[i]) + zero) * scal;
    local_min = std::min(local_min, value);
    local_max = std::max(local_max, value);
  }
  this->quant_mutex_.lock();
  this->observed_min_ = std::min(local_min, this->observed_min_);
  this->observed_max_ = std::max(local_max, this->observed_max_);
  this->quant_mutex_.unlock();
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Observe_out_cpu(size_t n, const void* data) {
  Observe_out_cpu(n, static_cast<const MOtype*>(data));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Observe_out_cpu(size_t n, const MOtype* data) {
  if (mode_ == CAFFE_QUANT_PASSIVE) {
    return;
  }
  double local_min = type_max_val<double>();
  double local_max = type_min_val<double>();
  double scal = out_scale_val();
  double zero = out_zero();
  for (size_t i = 0; i < n; ++i) {
    double value = (static_cast<double>(data[i]) + zero) * scal;
    local_min = std::min(local_min, value);
    local_max = std::max(local_max, value);
  }
  this->quant_mutex_.lock();
  this->observed_min_ = std::min(local_min, this->observed_min_);
  this->observed_max_ = std::max(local_max, this->observed_max_);
  this->quant_mutex_.unlock();
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::GenerateKernels() {
  this->program_ = this->device_->CreateProgram();
  stringstream ss;
  ss << this->program_->setup();
  ss << this->program_->template define_type<MItype>("MItype");
  ss << this->program_->template define_type<MOtype>("MOtype");
  ss << this->program_->template define_vector_type<MItype>("MItype", 0, 4);
  ss << this->program_->template define_vector_type<MOtype>("MOtype", 0, 4);


  // Quantizer forward
  {
    KernelArgs args;
    args.push_back(this->program_->template create_kernel_arg<uint_tp>("n",
                                                             KERNEL_ARG_CONST));
    args.push_back(this->program_->template create_kernel_arg<MItype>("in",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST |
               KERNEL_ARG_MEM_OFFSET));
    args.push_back(this->program_->template create_kernel_arg<MOtype>("out",
          KERNEL_ARG_MEM_OFFSET | KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM));
    if (fw_scale_before_cast()) {
      typedef typename std::conditional<float_is_same<MItype>::value, MItype,
              typename std::conditional<sizeof(MItype) == 1, int16_t,
              typename std::conditional<sizeof(MItype) == 2, int32_t, int64_t>
                                                  ::type>::type>::type Difftype;

      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                     "scal", KERNEL_ARG_CONST));
      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                  "in_zero", KERNEL_ARG_CONST));
      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                 "out_zero", KERNEL_ARG_CONST));
      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                  "min_out", KERNEL_ARG_CONST));
      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                  "max_out", KERNEL_ARG_CONST));
    } else {
      typedef typename std::conditional<float_is_same<MOtype>::value, MOtype,
              typename std::conditional<sizeof(MOtype) == 1, int16_t,
              typename std::conditional<sizeof(MOtype) == 2, int32_t, int64_t>
                                                  ::type>::type>::type Difftype;

      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                     "scal", KERNEL_ARG_CONST));
      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                  "in_zero", KERNEL_ARG_CONST));
      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                 "out_zero", KERNEL_ARG_CONST));
      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                  "min_out", KERNEL_ARG_CONST));
      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                  "max_out", KERNEL_ARG_CONST));
    }
    ss << this->program_->function("quantizer_forward", args);
    ss << this->program_->kernel_loop("uint_tp", "i", "n");
    ss << fw_gpu_term(0, "in[i]", "out[i]", "scal",
                      "in_zero", "out_zero", "min_out", "max_out") << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  // Quantizer backward
  {
    KernelArgs args;
    args.push_back(this->program_->template create_kernel_arg<uint_tp>("n",
                                                             KERNEL_ARG_CONST));
    args.push_back(this->program_->template create_kernel_arg<MOtype>("in",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST |
               KERNEL_ARG_MEM_OFFSET));
    args.push_back(this->program_->template create_kernel_arg<MItype>("out",
          KERNEL_ARG_MEM_OFFSET | KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM));
    if (bw_scale_before_cast()) {
      typedef typename std::conditional<float_is_same<MOtype>::value, MOtype,
              typename std::conditional<sizeof(MOtype) == 1, int16_t,
              typename std::conditional<sizeof(MOtype) == 2, int32_t, int64_t>
                                                  ::type>::type>::type Difftype;

      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                     "scal", KERNEL_ARG_CONST));
      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                  "in_zero", KERNEL_ARG_CONST));
      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                 "out_zero", KERNEL_ARG_CONST));
      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                  "min_out", KERNEL_ARG_CONST));
      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                  "max_out", KERNEL_ARG_CONST));
    } else {
      typedef typename std::conditional<float_is_same<MItype>::value, MItype,
              typename std::conditional<sizeof(MItype) == 1, int16_t,
              typename std::conditional<sizeof(MItype) == 2, int32_t, int64_t>
                                                  ::type>::type>::type Difftype;

      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                     "scal", KERNEL_ARG_CONST));
      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                  "in_zero", KERNEL_ARG_CONST));
      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                 "out_zero", KERNEL_ARG_CONST));
      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                  "min_out", KERNEL_ARG_CONST));
      args.push_back(this->program_->template create_kernel_arg<Difftype>(
                                                  "max_out", KERNEL_ARG_CONST));
    }
    ss << this->program_->function("quantizer_backward", args);
    ss << this->program_->kernel_loop("uint_tp", "i", "n");
    ss << bw_gpu_term(0, "in[i]", "out[i]", "scal",
                      "in_zero", "out_zero", "min_out", "max_out") << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  // Observe in
  {
    KernelArgs args;
    args.push_back(this->program_->template create_kernel_arg<uint_tp>("n",
                                                             KERNEL_ARG_CONST));
    args.push_back(this->program_->template create_kernel_arg<MItype>("data",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST |
               KERNEL_ARG_MEM_OFFSET));
    args.push_back(this->program_->template create_kernel_arg<float>("scal",
                                                             KERNEL_ARG_CONST));
    args.push_back(this->program_->template create_kernel_arg<float>(
                                               "zero", KERNEL_ARG_CONST));

    args.push_back(this->program_->template create_kernel_arg<float>(
        "inter_min", KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->program_->template create_kernel_arg<float>(
        "inter_max", KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM));
    ss << this->program_->function("quantizer_observe_in", args);
    ss << this->program_->local_mem("float", "local_min[" +
                   std::to_string(flp2(this->device_->workgroup_size(0))) + "]")
                       << ";" << std::endl;
    ss << this->program_->local_mem("float", "local_max[" +
                   std::to_string(flp2(this->device_->workgroup_size(0))) + "]")
                       << ";" << std::endl;
    ss << "local_min[" << this->program_->local_id(0) << "] = FLT_MAX;"
       << std::endl;
    ss << "local_max[" << this->program_->local_id(0) << "] = -FLT_MAX;"
       << std::endl;
    ss << this->program_->kernel_loop("uint_tp", "i", "n");
    ss << "float value = ((float)(data[i]) - zero) * scal;"
       << std::endl;
    ss << "if (local_min[" << this->program_->local_id(0) << "] > "
       << "value) {" << std::endl;
    ss << "local_min[" << this->program_->local_id(0) << "] = value;"
       << std::endl;
    ss << "}" << std::endl;
    ss << "if (local_max[" << this->program_->local_id(0) << "] < "
       << "value) {" << std::endl;
    ss << "local_max[" << this->program_->local_id(0) << "] = value;"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;  // Kernel loop
    // Reduction
    ss << this->program_->local_barrier() << std::endl;
    ss << "uint_tp i = " << flp2(this->device_->workgroup_size(0)/2) << ";"
       << std::endl;
    ss << "while (i != 0) {" << std::endl;
    ss << "if(" << this->program_->local_id(0) << " < i) {" << std::endl;
    ss << "if (local_min[" << this->program_->local_id(0) << "] > "
       << "local_min[" << this->program_->local_id(0) << " + i]) {"
       << std::endl;
    ss << "local_min[" << this->program_->local_id(0) << "] = "
       << "local_min[" << this->program_->local_id(0) << " + i];" << std::endl;
    ss << "}" << std::endl;
    ss << "if (local_max[" << this->program_->local_id(0) << "] < "
       << "local_max[" << this->program_->local_id(0) << " + i]) {"
       << std::endl;
    ss << "local_max[" << this->program_->local_id(0) << "] = "
       << "local_max[" << this->program_->local_id(0) << " + i];" << std::endl;
    ss << "}" << std::endl;    ss << "}" << std::endl;
    ss << this->program_->local_barrier() << std::endl;
    ss << "i /= 2;" << std::endl;
    ss << "}" << std::endl;  // while (i != 0)
    // Write partially reduced output
    ss << "if (" << this->program_->local_id(0) << " == 0 ) {" << std::endl;
    ss << "inter_min[" << this->program_->group_id(0) << "] = local_min[0];"
       << std::endl;
    ss << "inter_max[" << this->program_->group_id(0) << "] = local_max[0];"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  // Observe out
  {
    KernelArgs args;
    args.push_back(this->program_->template create_kernel_arg<uint_tp>("n",
                                                             KERNEL_ARG_CONST));
    args.push_back(this->program_->template create_kernel_arg<MOtype>("data",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST |
               KERNEL_ARG_MEM_OFFSET));
    args.push_back(this->program_->template create_kernel_arg<float>("scal",
                                                             KERNEL_ARG_CONST));
    args.push_back(this->program_->template create_kernel_arg<float>(
                                               "zero", KERNEL_ARG_CONST));

    args.push_back(this->program_->template create_kernel_arg<float>(
        "inter_min", KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->program_->template create_kernel_arg<float>(
        "inter_max", KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM));
    ss << this->program_->function("quantizer_observe_out", args);
    ss << this->program_->local_mem("float", "local_min[" +
                   std::to_string(flp2(this->device_->workgroup_size(0))) + "]")
                       << ";" << std::endl;
    ss << this->program_->local_mem("float", "local_max[" +
                   std::to_string(flp2(this->device_->workgroup_size(0))) + "]")
                       << ";" << std::endl;
    ss << "local_min[" << this->program_->local_id(0) << "] = FLT_MAX;"
        << std::endl;
    ss << "local_max[" << this->program_->local_id(0) << "] = -FLT_MAX;"
       << std::endl;
    ss << this->program_->kernel_loop("uint_tp", "i", "n");
    ss << "float value = ((float)(data[i]) - zero) * scal;"
       << std::endl;
    ss << "if (local_min[" << this->program_->local_id(0) << "] > "
       << "value) {" << std::endl;
    ss << "local_min[" << this->program_->local_id(0) << "] = value;"
       << std::endl;
    ss << "}" << std::endl;
    ss << "if (local_max[" << this->program_->local_id(0) << "] < "
       << "value) {" << std::endl;
    ss << "local_max[" << this->program_->local_id(0) << "] = value;"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;  // Kernel loop
    // Reduction
    ss << this->program_->local_barrier() << std::endl;
    ss << "uint_tp i = " << flp2(this->device_->workgroup_size(0)/2) << ";"
       << std::endl;
    ss << "while (i != 0) {" << std::endl;
    ss << "if(" << this->program_->local_id(0) << " < i) {" << std::endl;
    ss << "if (local_min[" << this->program_->local_id(0) << "] > "
       << "local_min[" << this->program_->local_id(0) << " + i]) {"
       << std::endl;
    ss << "local_min[" << this->program_->local_id(0) << "] = "
       << "local_min[" << this->program_->local_id(0) << " + i];" << std::endl;
    ss << "}" << std::endl;
    ss << "if (local_max[" << this->program_->local_id(0) << "] < "
       << "local_max[" << this->program_->local_id(0) << " + i]) {"
       << std::endl;
    ss << "local_max[" << this->program_->local_id(0) << "] = "
       << "local_max[" << this->program_->local_id(0) << " + i];" << std::endl;
    ss << "}" << std::endl;    ss << "}" << std::endl;
    ss << this->program_->local_barrier() << std::endl;
    ss << "i /= 2;" << std::endl;
    ss << "}" << std::endl;  // while (i != 0)
    // Write partially reduced output
    ss << "if (" << this->program_->local_id(0) << " == 0 ) {" << std::endl;
    ss << "inter_min[" << this->program_->group_id(0) << "] = local_min[0];"
       << std::endl;
    ss << "inter_max[" << this->program_->group_id(0) << "] = local_max[0];"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  this->program_->set_source(ss.str());
  this->program_->Compile(true, true);
  program_ready_ = true;
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Forward_gpu(size_t n, vptr<const MItype> input,
                         vptr<MOtype> output) {
  this->quant_mutex_.lock();
  if (!program_ready_) {
    this->GenerateKernels();
  }
  this->quant_mutex_.unlock();


  shared_ptr<DeviceKernel> kernel =
                                 this->program_->GetKernel("quantizer_forward");

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);

  uint_tp n_arg = n;
  kernel->add_arg(&n_arg);
  kernel->add_arg(&input);
  kernel->add_arg(&output);
  if (fw_scale_before_cast()) {
    typedef typename std::conditional<float_is_same<MItype>::value, MItype,
            typename std::conditional<sizeof(MItype) == 1, int16_t,
            typename std::conditional<sizeof(MItype) == 2, int32_t, int64_t>
                                                  ::type>::type>::type Difftype;

    const Difftype scal = static_cast<Difftype>(fw_scale_before_cast_val());
    const Difftype in_zero = static_cast<Difftype>(this->in_vals_.zero);
    const Difftype out_zero = static_cast<Difftype>(this->out_vals_.zero);
    const Difftype min_out = static_cast<Difftype>(this->out_vals_.min);
    const Difftype max_out = static_cast<Difftype>(this->out_vals_.max);
    kernel->add_arg(&scal);
    kernel->add_arg(&in_zero);
    kernel->add_arg(&out_zero);
    kernel->add_arg(&min_out);
    kernel->add_arg(&max_out);
    kernel->Execute(group, local);
  } else {
    typedef typename std::conditional<float_is_same<MOtype>::value, MOtype,
            typename std::conditional<sizeof(MOtype) == 1, int16_t,
            typename std::conditional<sizeof(MOtype) == 2, int32_t, int64_t>
                                                  ::type>::type>::type Difftype;

    const Difftype scal = static_cast<Difftype>(fw_scale_after_cast_val());
    const Difftype in_zero = static_cast<Difftype>(this->in_vals_.zero);
    const Difftype out_zero = static_cast<Difftype>(this->out_vals_.zero);
    const Difftype min_out = static_cast<Difftype>(this->out_vals_.min);
    const Difftype max_out = static_cast<Difftype>(this->out_vals_.max);
    kernel->add_arg(&scal);
    kernel->add_arg(&in_zero);
    kernel->add_arg(&out_zero);
    kernel->add_arg(&min_out);
    kernel->add_arg(&max_out);
    kernel->Execute(group, local);
  }
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Forward_gpu(size_t n, vptr<const void> input,
                         vptr<void> output) {
  this->Forward_gpu(n, vptr<const MItype>(input), vptr<MOtype>(output));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Forward_gpu(Blob<MItype>* input,
                                            Blob<MOtype>* output,
                                            bool fw_data,
                                            bool fw_diff) {
  CHECK_EQ(input->count(), output->count());
  if (fw_data) {
    this->Forward_gpu(input->count(),
                      input->gpu_data(),
                      output->mutable_gpu_data());
  }
  if (fw_diff) {
    this->Forward_gpu(input->count(),
                      input->gpu_diff(),
                      output->mutable_gpu_diff());
  }
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Backward_gpu(size_t n, vptr<const MOtype> input,
                         vptr<MItype> output) {
  this->quant_mutex_.lock();
  if (!program_ready_) {
    this->GenerateKernels();
  }
  this->quant_mutex_.unlock();


  shared_ptr<DeviceKernel> kernel =
                                this->program_->GetKernel("quantizer_backward");

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);

  uint_tp n_arg = n;
  kernel->add_arg(&n_arg);
  kernel->add_arg(&input);
  kernel->add_arg(&output);
  if (bw_scale_before_cast()) {
    typedef typename std::conditional<float_is_same<MOtype>::value, MOtype,
            typename std::conditional<sizeof(MOtype) == 1, int16_t,
            typename std::conditional<sizeof(MOtype) == 2, int32_t, int64_t>
                                                  ::type>::type>::type Difftype;

    const Difftype scal = static_cast<Difftype>(bw_scale_before_cast_val());
    const Difftype in_zero = static_cast<Difftype>(this->out_vals_.zero);
    const Difftype out_zero = static_cast<Difftype>(this->in_vals_.zero);
    const Difftype min_out = static_cast<Difftype>(this->in_vals_.min);
    const Difftype max_out = static_cast<Difftype>(this->in_vals_.max);
    kernel->add_arg(&scal);
    kernel->add_arg(&in_zero);
    kernel->add_arg(&out_zero);
    kernel->add_arg(&min_out);
    kernel->add_arg(&max_out);
    kernel->Execute(group, local);
  } else {
    typedef typename std::conditional<float_is_same<MItype>::value, MItype,
            typename std::conditional<sizeof(MItype) == 1, int16_t,
            typename std::conditional<sizeof(MItype) == 2, int32_t, int64_t>
                                                  ::type>::type>::type Difftype;

    const Difftype scal = static_cast<Difftype>(bw_scale_after_cast_val());
    const Difftype in_zero = static_cast<Difftype>(this->out_vals_.zero);
    const Difftype out_zero = static_cast<Difftype>(this->in_vals_.zero);
    const Difftype min_out = static_cast<Difftype>(this->in_vals_.min);
    const Difftype max_out = static_cast<Difftype>(this->in_vals_.max);
    kernel->add_arg(&scal);
    kernel->add_arg(&in_zero);
    kernel->add_arg(&out_zero);
    kernel->add_arg(&min_out);
    kernel->add_arg(&max_out);
    kernel->Execute(group, local);
  }
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Backward_gpu(size_t n, vptr<const void> input,
                         vptr<void> output) {
  this->Backward_gpu(n, vptr<const MOtype>(input), vptr<MItype>(output));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Backward_gpu(Blob<MOtype>* input,
                                             Blob<MItype>* output,
                                             bool bw_data,
                                             bool bw_diff) {
  CHECK_EQ(input->count(), output->count());
  if (bw_data) {
    this->Backward_gpu(input->count(),
                       input->gpu_data(),
                       output->mutable_gpu_data());
  }
  if (bw_diff) {
    this->Backward_gpu(input->count(),
                       input->gpu_diff(),
                       output->mutable_gpu_diff());
  }
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Observe_in_gpu(size_t n,
                                               vptr<const void> data) {
  Observe_in_gpu(n, vptr<const MItype>(data));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Observe_in_gpu(size_t n,
                                               vptr<const MItype> data) {
  if (mode_ == CAFFE_QUANT_PASSIVE) {
    return;
  }
  this->quant_mutex_.lock();
  if (!program_ready_) {
    this->GenerateKernels();
  }
  this->quant_mutex_.unlock();

  float scal = in_scale_val();
  float zero = in_zero();

  vector<size_t> local(1, flp2(this->device_->workgroup_size(0)));
  vector<size_t> group(1, (n - 1) / local[0] + 1);

  int_tp min_buffer_lock_id = -1;
  int_tp max_buffer_lock_id = -1;

  shared_ptr<Blob<float> > min_blob =
      this->device_->template Buffer<float>(vector<int_tp>(1, group[0]),
                                            &min_buffer_lock_id);
  shared_ptr<Blob<float> > max_blob =
      this->device_->template Buffer<float>(vector<int_tp>(1, group[0]),
                                            &max_buffer_lock_id);

  vptr<float> min_gpu_data = min_blob->mutable_gpu_data();
  this->device_->template set<float>(group[0], type_max_val<float>(),
                                     min_gpu_data);
  vptr<float> max_gpu_data = max_blob->mutable_gpu_data();
  this->device_->template set<float>(group[0], type_min_val<float>(),
                                     max_gpu_data);

  shared_ptr<DeviceKernel> kernel =
                              this->program_->GetKernel("quantizer_observe_in");

  uint_tp n_arg = n;
  kernel->add_arg(&n_arg);
  kernel->add_arg(&data);
  kernel->add_arg(&scal);
  kernel->add_arg(&zero);
  kernel->add_arg(&min_gpu_data);
  kernel->add_arg(&max_gpu_data);
  kernel->Execute(group, local);

  const float* min_cpu_data = min_blob->cpu_data();
  const float* max_cpu_data = max_blob->cpu_data();

  this->quant_mutex_.lock();
  for (size_t i = 0; i < group[0]; ++i) {
    observed_min_ = std::min(static_cast<double>(min_cpu_data[i]),
                             observed_min_);
    observed_max_ = std::max(static_cast<double>(max_cpu_data[i]),
                             observed_max_);
  }
  this->quant_mutex_.unlock();

  this->device_->unlock_buffer(&min_buffer_lock_id);
  this->device_->unlock_buffer(&max_buffer_lock_id);
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Observe_out_gpu(size_t n,
                                                vptr<const void> data) {
  Observe_out_gpu(n, vptr<const MOtype>(data));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Observe_out_gpu(size_t n,
                                                vptr<const MOtype> data) {
  if (mode_ == CAFFE_QUANT_PASSIVE) {
    return;
  }
  this->quant_mutex_.lock();
  if (!program_ready_) {
    this->GenerateKernels();
  }
  this->quant_mutex_.unlock();

  float scal = out_scale_val();
  float zero = out_zero();

  vector<size_t> local(1, flp2(this->device_->workgroup_size(0)));
  vector<size_t> group(1, (n - 1) / local[0] + 1);

  int_tp min_buffer_lock_id = -1;
  int_tp max_buffer_lock_id = -1;

  shared_ptr<Blob<float> > min_blob =
      this->device_->template Buffer<float>(vector<int_tp>(1, group[0]),
                                            &min_buffer_lock_id);
  shared_ptr<Blob<float> > max_blob =
      this->device_->template Buffer<float>(vector<int_tp>(1, group[0]),
                                            &max_buffer_lock_id);

  vptr<float> min_gpu_data = min_blob->mutable_gpu_data();
  this->device_->template set<float>(group[0], type_max_val<float>(),
                                     min_gpu_data);
  vptr<float> max_gpu_data = max_blob->mutable_gpu_data();
  this->device_->template set<float>(group[0], type_min_val<float>(),
                                     max_gpu_data);

  shared_ptr<DeviceKernel> kernel =
                             this->program_->GetKernel("quantizer_observe_out");

  uint_tp n_arg = n;
  kernel->add_arg(&n_arg);
  kernel->add_arg(&data);
  kernel->add_arg(&scal);
  kernel->add_arg(&zero);
  kernel->add_arg(&min_gpu_data);
  kernel->add_arg(&max_gpu_data);
  kernel->Execute(group, local);

  const float* min_cpu_data = min_blob->cpu_data();
  const float* max_cpu_data = max_blob->cpu_data();

  this->quant_mutex_.lock();
  for (size_t i = 0; i < group[0]; ++i) {
    observed_min_ = std::min(static_cast<double>(min_cpu_data[i]),
                             observed_min_);
    observed_max_ = std::max(static_cast<double>(max_cpu_data[i]),
                             observed_max_);
  }
  this->quant_mutex_.unlock();

  this->device_->unlock_buffer(&min_buffer_lock_id);
  this->device_->unlock_buffer(&max_buffer_lock_id);
}

INSTANTIATE_CLASS_2T(Quantizer, PROTO_TYPES, PROTO_TYPES)


}  // namespace caffe
