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

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::update_param(const QuantizerParameter& param) {
  quant_mutex_.lock();
  this->param_ = param;
  this->param_.set_input_data_type(proto_data_type<MItype>());
  this->param_.set_output_data_type(proto_data_type<MOtype>());
  this->program_ready_ = false;
  this->index_ = param_.index();
  this->mode_ = param_.mode();

  // Load estimated max/min values
  if (this->param_.has_observed_min()) {
    this->observed_min_ = param_.observed_min();
  }
  if (this->param_.has_observed_max()) {
    this->observed_max_ = param_.observed_max();
  }

  // Set floating point (symmetric) maximum and minimum
  this->flt_min_ = std::min(this->observed_min_, -this->observed_max_);
  this->flt_max_ = std::max(this->observed_max_, -this->observed_min_);

  if (this->param_.has_min_in()) {
    this->min_in_ = param_.min_in();
  } else {
    if (is_float_type<MItype>()) {
      this->min_in_ = this->flt_min_;
    }
    if (is_signed_integer_type<MItype>()) {
      this->min_in_ = type_min_val<MItype>();
    }
  }
  if (this->param_.has_max_in()) {
    this->max_in_ = param_.max_in();
  } else {
    if (is_float_type<MItype>()) {
      this->max_in_ = this->flt_max_;
    }
    if (is_signed_integer_type<MItype>()) {
      this->max_in_ = type_max_val<MItype>();
    }
  }

  if (this->param_.has_min_out()) {
    this->min_out_ = param_.min_out();
  } else {
    if (is_float_type<MOtype>()) {
      this->min_out_ = this->flt_min_;
    }
    if (is_signed_integer_type<MOtype>()) {
      this->min_out_ = type_min_val<MOtype>();
    }
  }
  if (this->param_.has_max_out()) {
    this->max_out_ = param_.max_out();
  } else {
    if (is_float_type<MOtype>()) {
      this->max_out_ = this->flt_max_;
    }
    if (is_signed_integer_type<MOtype>()) {
      this->max_out_ = type_max_val<MOtype>();
    }
  }
  quant_mutex_.unlock();
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::init() {
  this->program_ = this->device_->CreateProgram();

  this->mode_ = PASSIVE;

  this->observed_min_ = type_max_val<double>();
  this->observed_max_ = type_min_val<double>();

  this->flt_min_ = type_min_val<double>();
  this->flt_max_ = type_max_val<double>();
  this->min_in_ = type_min_val<MItype>();
  this->max_in_ = type_max_val<MItype>();
  this->min_out_ = type_min_val<MOtype>();
  this->max_out_ = type_max_val<MOtype>();
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
      if (this->min_in_ == this->min_out_ && this->max_in_ == this->max_out_) {
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
Quantizer<MItype, MOtype>::Quantizer(const QuantizerParameter& param)
  : QuantizerBase(param) {
  init();
  update_param(param_);
}

template<typename MItype, typename MOtype>
Quantizer<MItype, MOtype>::Quantizer(Device* dev_ptr)
  : QuantizerBase(dev_ptr) {
  init();
  QuantizerParameter param;
  param.set_device(dev_ptr->id());
  param.set_input_data_type(proto_data_type<MItype>());
  param.set_output_data_type(proto_data_type<MOtype>());
  param.set_index(0);
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
    if (std::is_same<MItype, MOtype>::value) {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        output[i] = input[i];
      }
    } else {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        output[i] = static_cast<MOtype>(input[i]);
      }
    }
    return;
  }

  if (fw_scale_before_cast()) {
    MItype scal = fw_scale_before_cast_val();
    if (fw_scale_divide()) {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        output[i] = static_cast<MOtype>(input[i] / scal);
      }
    } else {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        output[i] = static_cast<MOtype>(input[i] * scal);
      }
    }
  } else {
    MOtype scal = fw_scale_after_cast_val();
    if (fw_scale_divide()) {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        output[i] = static_cast<MOtype>(input[i]) / scal;
      }
    } else {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        output[i] = static_cast<MOtype>(input[i]) * scal;
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
    if (std::is_same<MItype, MOtype>::value) {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        output[i] = input[i];
      }
    } else {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        output[i] = static_cast<MItype>(input[i]);
      }
    }
    return;
  }

  if (bw_scale_before_cast()) {
    MOtype scal = bw_scale_before_cast_val();
    if (bw_scale_divide()) {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        output[i] = static_cast<MItype>(input[i] / scal);
      }
    } else {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        output[i] = static_cast<MItype>(input[i] * scal);
      }
    }
  } else {
    MItype scal = bw_scale_after_cast_val();
    if (bw_scale_divide()) {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        output[i] = static_cast<MItype>(input[i]) / scal;
      }
    } else {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        output[i] = static_cast<MItype>(input[i]) * scal;
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
    val = std::max(std::abs(max_in_), std::abs(min_in_))
      / std::max(std::abs(max_out_), std::abs(min_out_));
  } else {
    val = std::max(std::abs(max_out_), std::abs(min_out_)
      / std::max(std::abs(max_in_), std::abs(min_in_)));
  }
  return static_cast<MItype>(val);
}

template<typename MItype, typename MOtype>
MOtype Quantizer<MItype, MOtype>::fw_scale_after_cast_val() const {
  double val = 0.0;
  if (this->fw_scale_divide()) {
    val = std::max(std::abs(max_in_), std::abs(min_in_))
        / std::max(std::abs(max_out_), std::abs(min_out_));
  } else {
    val = std::max(std::abs(max_out_), std::abs(min_out_)
        / std::max(std::abs(max_in_), std::abs(min_in_)));
  }
  return static_cast<MOtype>(val);
}

template<typename MItype, typename MOtype>
MOtype Quantizer<MItype, MOtype>::bw_scale_before_cast_val() const {
  double val = 0.0;
  if (this->fw_scale_divide()) {
    val = std::max(std::abs(max_out_), std::abs(min_out_)
        / std::max(std::abs(max_in_), std::abs(min_in_)));
  } else {
    val = std::max(std::abs(max_in_), std::abs(min_in_))
        / std::max(std::abs(max_out_), std::abs(min_out_));
  }
  return static_cast<MOtype>(val);
}

template<typename MItype, typename MOtype>
MItype Quantizer<MItype, MOtype>::bw_scale_after_cast_val() const {
  double val = 0.0;
  if (this->fw_scale_divide()) {
    val = std::max(std::abs(max_out_), std::abs(min_out_)
        / std::max(std::abs(max_in_), std::abs(min_in_)));
  } else {
    val = std::max(std::abs(max_in_), std::abs(min_in_))
        / std::max(std::abs(max_out_), std::abs(min_out_));
  }
  return static_cast<MItype>(val);
}


template<typename MItype, typename MOtype>
string Quantizer<MItype, MOtype>::fw_scale_term(int_tp vec_len,
                                                 string scale_var,
                                                 string src_val) const {
  string tmp = "";
  if (!needs_quantization()) {
    if (std::is_same<MItype, MOtype>::value) {
      return src_val;
    }
  } else {
    if (this->fw_scale_divide()) {
      tmp = "/ " + scale_var;
    } else {
      tmp = "* " + scale_var;
    }
  }
  if (this->fw_scale_before_cast()) {
    return "(" + program_->template convert_type<MOtype>(vec_len,
                                                         src_val + tmp) + ")";
  } else {
    return "(" + program_->template convert_type<MOtype>(vec_len,
                                                         src_val) + tmp + ")";
  }
}

template<typename MItype, typename MOtype>
string Quantizer<MItype, MOtype>::bw_scale_term(int_tp vec_len,
                                                 string scale_var,
                                                 string src_val) const {
  string tmp = "";
  if (!needs_quantization()) {
    if (std::is_same<MItype, MOtype>::value) {
      return src_val;
    }
  } else {
    if (this->bw_scale_divide()) {
      tmp = "/ " + scale_var;
    } else {
      tmp = "* " + scale_var;
    }
  }
  if (this->bw_scale_before_cast()) {
    return "(" + program_->template convert_type<MItype>(vec_len,
                                                         src_val + tmp) + ")";
  } else {
    return "(" + program_->template convert_type<MItype>(vec_len,
                                                         src_val) + tmp + ")";
  }
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Observe_in_cpu(size_t n, const void* data) {
  Observe_in_cpu(n, static_cast<const MItype*>(data));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Observe_in_cpu(size_t n,
                                                 const MItype* data) {
  if (mode_ == PASSIVE) {
    return;
  }
  double local_min = type_max_val<double>();
  double local_max = type_min_val<double>();
  double scal = std::max(std::abs(max_in_), std::abs(min_in_))
              / std::max(std::abs(flt_max_), std::abs(flt_min_));
  for (size_t i = 0; i < n; ++i) {
    double value;
    if (is_signed_integer_type<MItype>()) {
      value = static_cast<double>(data[i])/scal;
    } else {
      value = data[i];
    }
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
void Quantizer<MItype, MOtype>::Observe_out_cpu(size_t n,
                                                  const MOtype* data) {
  if (mode_ == PASSIVE) {
    return;
  }
  double local_min = type_max_val<double>();
  double local_max = type_min_val<double>();
  double scal = std::max(std::abs(max_in_), std::abs(min_in_))
              / std::max(std::abs(flt_max_), std::abs(flt_min_));
  for (size_t i = 0; i < n; ++i) {
    double value;
    if (is_signed_integer_type<MItype>()) {
      value = static_cast<double>(data[i])/scal;
    } else {
      value = data[i];
    }
    local_min = std::min(local_min, value);
    local_max = std::max(local_max, value);
  }
  this->quant_mutex_.lock();
  this->observed_min_ = std::min(local_min, this->observed_min_);
  this->observed_max_ = std::max(local_max, this->observed_max_);
  this->quant_mutex_.unlock();
}

INSTANTIATE_CLASS_2T(Quantizer, PROTO_TYPES, PROTO_TYPES)


}  // namespace caffe
