#include "caffe/quantizer.hpp"
#include "caffe/util/type_utils.hpp"

namespace caffe {

QuantizerBase::QuantizerBase(QuantizerParameter& param)
  : quant_param_(param), program_ready_(false) {
  device_ = Caffe::GetDevice(quant_param_.device(), true);
  program_ = device_->CreateProgram();
}

QuantizerMode QuantizerBase::get_mode() const {
  return mode_;
}

void QuantizerBase::set_mode(QuantizerMode mode) {
  mode_ = mode;
  program_ready_ = false;
}

template<typename MItype, typename MOtype>
Quantizer<MItype, MOtype>::Quantizer(QuantizerParameter& param)
  : QuantizerBase(param) {
  min_in_ = type_min_val<MItype>();
  max_in_ = type_max_val<MItype>();
  min_out_ = type_min_val<MOtype>();
  max_out_ = type_max_val<MOtype>();
  if (std::is_same<MItype, MOtype>::value) {
    mode_ = QUANTIZER_MODE_PASSIVE;
  } else {
    if (is_float_type<MItype>() && is_float_type<MOtype>()) {
      mode_ = QUANTIZER_MODE_PASSIVE;
    } else {
      mode_ = QUANTIZER_MODE_ACTIVE;
    }
  }
}



template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Forward_cpu(Blob<MItype>* input,
                                            Blob<MOtype>* output,
                                            bool fw_data, bool fw_diff) {
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
   size_t n, const void* input, void* output) {
  this->Forward_cpu(n,
                    static_cast<const MItype*>(input),
                    static_cast<MOtype*>(output));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Forward_cpu(
   size_t n, const MItype* input, MOtype* output) {
  CHECK(input);
  CHECK(output);

  if (mode_ == QUANTIZER_MODE_PASSIVE || mode_ == QUANTIZER_MODE_OBSERVE) {
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
                                             bool bw_data, bool bw_diff) {
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
   size_t n, const void* input, void* output) {
  this->Backward_cpu(n,
                     static_cast<const MOtype*>(input),
                     static_cast<MItype*>(output));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Backward_cpu(
   size_t n, const MOtype* input, MItype* output) {
  CHECK(input);
  CHECK(output);

  if (mode_ == QUANTIZER_MODE_PASSIVE || mode_ == QUANTIZER_MODE_OBSERVE) {
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
    val = max(abs(max_in_), abs(min_in_)) / max(abs(max_out_), abs(min_out_));
  } else {
    val = max(abs(max_out_), abs(min_out_) / max(abs(max_in_), abs(min_in_)));
  }
  return static_cast<MItype>(val);
}

template<typename MItype, typename MOtype>
MOtype Quantizer<MItype, MOtype>::fw_scale_after_cast_val() const {
  double val = 0.0;
  if (this->fw_scale_divide()) {
    val = max(abs(max_in_), abs(min_in_)) / max(abs(max_out_), abs(min_out_));
  } else {
    val = max(abs(max_out_), abs(min_out_) / max(abs(max_in_), abs(min_in_)));
  }
  return static_cast<MOtype>(val);
}

template<typename MItype, typename MOtype>
MOtype Quantizer<MItype, MOtype>::bw_scale_before_cast_val() const {
  double val = 0.0;
  if (this->fw_scale_divide()) {
    val = max(abs(max_out_), abs(min_out_) / max(abs(max_in_), abs(min_in_)));
  } else {
    val = max(abs(max_in_), abs(min_in_)) / max(abs(max_out_), abs(min_out_));
  }
  return static_cast<MOtype>(val);
}

template<typename MItype, typename MOtype>
MItype Quantizer<MItype, MOtype>::bw_scale_after_cast_val() const {
  double val = 0.0;
  if (this->fw_scale_divide()) {
    val = max(abs(max_out_), abs(min_out_) / max(abs(max_in_), abs(min_in_)));
  } else {
    val = max(abs(max_in_), abs(min_in_)) / max(abs(max_out_), abs(min_out_));
  }
  return static_cast<MItype>(val);
}


template<typename MItype, typename MOtype>
string  Quantizer<MItype, MOtype>::fw_scale_term(int_tp vec_len,
                                                 string scale_var,
                                                 string src_val) const {
  string tmp = "";
  if (mode_ == QUANTIZER_MODE_PASSIVE || mode_ == QUANTIZER_MODE_OBSERVE) {
    if (std::is_same<MItype, MOtype>::value) {
      return src_val;
    }
  } else {
    if (this->fw_scale_divide()) {
      tmp = "/ scale_var";
    } else {
      tmp = "* scale_var";
    }
  }
  if (this->fw_scale_before_cast()) {
    return "(" + program_->convert_type(vec_len, src_val + tmp) + ")";
  } else {
    return "(" + program_->convert_type(vec_len, src_val) + tmp + ")";
  }
}

template<typename MItype, typename MOtype>
string  Quantizer<MItype, MOtype>::bw_scale_term(int_tp vec_len,
                                                 string scale_var,
                                                 string src_val) const {
  string tmp = "";
  if (mode_ == QUANTIZER_MODE_PASSIVE || mode_ == QUANTIZER_MODE_OBSERVE) {
    if (std::is_same<MItype, MOtype>::value) {
      return src_val;
    }
  } else {
    if (this->bw_scale_divide()) {
      tmp = "/ scale_var";
    } else {
      tmp = "* scale_var";
    }
  }
  if (this->bw_scale_before_cast()) {
    return "(" + program_->convert_type(vec_len, src_val + tmp) + ")";
  } else {
    return "(" + program_->convert_type(vec_len, src_val) + tmp + ")";
  }
}


INSTANTIATE_CLASS_2T(Quantizer, VARIANT_TYPES, VARIANT_TYPES)


}  // namespace caffe
