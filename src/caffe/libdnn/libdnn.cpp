#include <string>
#include <vector>
#include "caffe/common.hpp"

#ifdef USE_LIBDNN
#include "caffe/backend/device.hpp"
#include "caffe/libdnn/libdnn.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
LibDNN<Dtype, MItype, MOtype>::LibDNN(Device* dev_ptr)
  : dev_ptr_(dev_ptr), fast_unsafe_math_(false),
    program_(dev_ptr->CreateProgram()) {
};

template<typename Dtype, typename MItype, typename MOtype>
string LibDNN<Dtype, MItype, MOtype>::generate_gemm_core(
    shared_ptr<LibDNNTuner> tuner, bool dterm, LibDNNAccumulatePrecision prec) {
  stringstream ss;
  int vwm = tuner->get_param<int>("VWM");
  int vwn = tuner->get_param<int>("VWN");
  int rtsn = tuner->get_param<int>("workgroup_size_0");
  int rtsm = tuner->get_param<int>("workgroup_size_1");
  bool unroll = tuner->get_param<bool>("vector_unroll");

  // Temporary registers for a and b
  ss << "Dtype" << vwm << " Areg;" << std::endl;
  ss << "Dtype" << vwn << " Breg[WPTN/VWN];" << std::endl;

  // Loop over the values of a single tile
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int_tp kt = 0; kt < TSK; kt += TSK_UNROLL) {" << std::endl;
  ss << "#pragma unroll " << tuner->get_param<int>("TSK_UNROLL") << std::endl;
  ss << "for (int_tp ku = 0; ku < TSK_UNROLL; ++ku) {" << std::endl;
  ss << "int_tp k = kt + ku;" << std::endl;

  // Cache the values of Bsub in registers
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {" << std::endl;
  ss << "int_tp col = tidn + wn * VWN * RTSN;" << std::endl;
  for (int i = 0; i < vwn; ++i) {
    ss << "VEC_" << vwn << "_" << i << "(Breg[wn])"
       << " = Bsub[k][col + " << (i * rtsn)
       << "];" << std::endl;
  }
  ss << "}" << std::endl;

  // Perform the computation
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm = 0; wm < WPTM / VWM; ++wm) {" << std::endl;
  ss << "int_tp row = tidm + wm * VWM * RTSM;" << std::endl;
  for (int i = 0; i < vwm; ++i) {
    ss << "VEC_" << vwm << "_" << i << "(Areg)" << " = Asub[row + " << (i*rtsm)
       << "][k];" << std::endl;
  }
  if (dterm) {
    if (unroll) {
      for (int i = 0; i < vwm; ++i) {
        ss << "VEC_" << vwm << "_" << i << "(Dreg[wm]) " << "+= VEC_" << vwm
           << "_" << i << "(Areg) * v_bmul;" << std::endl;
      }
    } else {
      ss << "Dreg[wm] += Areg * v_bmul;" << std::endl;
    }
  }
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {" << std::endl;
  if (unroll) {
    for (int N = 0; N < vwn; ++N) {
      for (int M = 0; M < vwm; ++M) {
        ss << "VEC_" << vwn << "_" << N << "(Creg[wm * VWM + " << M << "][wn])"
           << " += VEC_" << vwm << "_" << M << "(Areg)" << " * VEC_" << vwn
           << "_" << N << "(Breg[wn]);" << std::endl;
      }
    }
  } else {
    for (int M = 0; M < vwm; ++M) {
      ss << "Creg[wm * VWM + " << M << "][wn]"
         << " += VEC_"<< vwm << "_" << M << "(Areg)" << " * (Breg[wn]);"
         << std::endl;
    }
  }
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Loop over a single tile
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

template<typename Dtype, typename MItype, typename MOtype>
string LibDNN<Dtype, MItype, MOtype>::generate_accreg_init(
    shared_ptr<LibDNNTuner> tuner, bool dterm, bool load, bool beta_term,
    LibDNNAccumulatePrecision prec) {
  stringstream ss;

  int vwm = tuner->get_param<int>("VWM");
  int vwn = tuner->get_param<int>("VWN");
  bool unroll = tuner->get_param<bool>("vector_unroll");

  string accreg_type = "Dtype";
  switch (prec) {
    case LIBDNN_ACCUMULATE_PREC_NATIVE:
      break;
    case LIBDNN_ACCUMULATE_PREC_8:
      accreg_type = program_->device_type_name<int8_t>();
      break;
    case LIBDNN_ACCUMULATE_PREC_16:
      accreg_type = program_->device_type_name<int16_t>();
      break;
    case LIBDNN_ACCUMULATE_PREC_32:
      accreg_type = program_->device_type_name<int32_t>();
      break;
    case LIBDNN_ACCUMULATE_PREC_64:
      accreg_type = program_->device_type_name<int64_t>();
      break;
    default:
      break;
  }

  string rescale = " rescale_in * ";
  if (is_integer_type<>())

  if (dterm) {
    ss << accreg_type << vwm << " Dreg[WPTM / VWM];" << std::endl;
  }
  ss << accreg_type << vwn << " Creg[WPTM][WPTN / VWN];" << std::endl;

  // Initialize the accumulation registers
  if (load) {
    // Load
    if (dterm) {
      ss << "#pragma unroll" << std::endl;
      ss << "for (int_tp wm = 0; wm < WPTM; ++wm) {" << std::endl;
      ss << "int_tp globalRow = offM + tidm + wm * RTSM;"
         << std::endl;
      ss << "((Dtype*)(&(Dreg[wm / VWM])))[wm %V WM] = "
          "Dptr[globalRow];" << std::endl;
      ss << "}" << std::endl;
    }
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wm = 0; wm < WPTM; ++wm) {" << std::endl;
    ss << "int_tp globalRow = offM + tidm + wm * RTSM;"
       << std::endl;
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wn = 0; wn < WPTN; ++wn) {" << std::endl;
    ss << "int_tp globalCol = offN + tidn + wn * RTSN;"
       << std::endl;
    ss << "if (globalRow < M && globalCol < N) {" << std::endl;
    ss << "((Dtype*)(&(Creg[wm][wn / VWN])))[wn % VWN] = ";
    if (beta_term) {
      ss << "beta * ";
    }
    ss << "Cptr[globalRow * N + globalCol];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  } else {
    // Zero init
    if (dterm) {
      ss << "#pragma unroll" << std::endl;
      ss << "for (int_tp wm = 0; wm < WPTM / VWM; ++wm) {" << std::endl;
      if (unroll) {
        for (int i = 0; i < vwm; ++i) {
          ss << "VEC_" << vwm << "_" << i << "(Dreg[wm]) = (Dtype)0;"
             << std::endl;
        }
      } else {
        ss << "Dreg[wm] = (Dtype)0;" << std::endl;
      }
      ss << "}" << std::endl;
    }
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wm = 0; wm < WPTM; ++wm) {" << std::endl;
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {" << std::endl;
    if (unroll) {
      for (int i = 0; i < vwn; ++i) {
        ss << "VEC_" << vwn << "_" << i << "(Creg[wm][wn]) = (Dtype)0;"
           << std::endl;
      }
    } else {
      ss << "Creg[wm][wn] = (Dtype)0;" << std::endl;
    }
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }
  return ss.str();
}

INSTANTIATE_CLASS_3T(LibDNN, (half_fp), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNN, (float), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNN, (double), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNN, (int8_t), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNN, (int16_t), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNN, (int32_t), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNN, (int64_t), VARIANT_TYPES, VARIANT_TYPES);

}  // namespace caffe

#endif  // USE_LIBDNN
