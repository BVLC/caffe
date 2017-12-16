#ifdef USE_LIBDNN

#include <string>
#include <vector>
#include "caffe/common.hpp"

#include "caffe/backend/device.hpp"
#include "caffe/libdnn/libdnn.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

LibDNNBase::LibDNNBase(Device* dev_ptr)
  : dev_ptr_(dev_ptr), fast_unsafe_math_(false),
    program_(dev_ptr->CreateProgram()) {
}

class LibDNNBase;

template<typename MItype, typename MOtype>
LibDNN<MItype, MOtype>::LibDNN(Device* dev_ptr)
  : LibDNNBase(dev_ptr) {
}

template<typename MItype, typename MOtype>
string LibDNN<MItype, MOtype>::generate_gemm_core(
    shared_ptr<LibDNNTuner> tuner, bool dterm, bool alpha_term,
    libdnnAccumulatePrecision_t prec) {
  stringstream ss;
  int vwm = tuner->get_param<int>("VWM");
  int vwn = tuner->get_param<int>("VWN");
  int rtsn = tuner->get_param<int>("workgroup_size_0");
  int rtsm = tuner->get_param<int>("workgroup_size_1");
  bool unroll = tuner->get_param<bool>("vector_unroll");

  string accreg_type = "MItype";
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

  // Temporary registers for a and b
  ss << "MItype" << vwm << " Areg;" << std::endl;
  ss << "MItype" << vwn << " Breg[WPTN/VWN];" << std::endl;

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
        ss << "VEC_" << vwm << "_" << i << "(Dreg[wm]) " << "+= ";
        if (prec != LIBDNN_ACCUMULATE_PREC_NATIVE) {
          ss << "(" << accreg_type << ")";
        }
        ss << "(VEC_" << vwm << "_" << i << "(Areg) * v_bmul);" << std::endl;
      }
    } else {
      ss << "Dreg[wm] += ";
      switch (prec) {
        case LIBDNN_ACCUMULATE_PREC_8:
          ss << this->program_->template convert_type<int8_t>(vwm,
                                                              "Areg * v_bmul");
          break;
        case LIBDNN_ACCUMULATE_PREC_16:
          ss << this->program_->template convert_type<int16_t>(vwm,
                                                               "Areg * v_bmul");
          break;
        case LIBDNN_ACCUMULATE_PREC_32:
          ss << this->program_->template convert_type<int32_t>(vwm,
                                                               "Areg * v_bmul");
          break;
        case LIBDNN_ACCUMULATE_PREC_64:
          ss << this->program_->template convert_type<int64_t>(vwm,
                                                               "Areg * v_bmul");
          break;
        case LIBDNN_ACCUMULATE_PREC_NATIVE:
        default:
          ss << "Areg * v_bmul;" << std::endl;
          break;
      }
    }
  }
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {" << std::endl;
  if (unroll) {
    for (int n = 0; n < vwn; ++n) {
      for (int m = 0; m < vwm; ++m) {
        ss << "VEC_" << vwn << "_" << n << "(Creg[wm * VWM + " << m << "][wn])"
           << " += ";
        if (prec != LIBDNN_ACCUMULATE_PREC_NATIVE) {
          ss << "(" << accreg_type << ")";
        }
        ss << "(VEC_" << vwm << "_" << m << "(Areg)" << " * VEC_" << vwn
           << "_" << n << "(Breg[wn]));" << std::endl;
      }
    }
  } else {
    for (int m = 0; m < vwm; ++m) {
      ss << "Creg[wm * VWM + " << m << "][wn] += ";
      stringstream src_term;
      if (alpha_term) {
        src_term << "(alpha *";
      } else {
        src_term << "(";
      }
      src_term << " * VEC_"<< vwm << "_" << m << "(Areg)"
               << " * (Breg[wn]));" << std::endl;
      switch (prec) {
        case LIBDNN_ACCUMULATE_PREC_8:
          ss << this->program_->template convert_type<int8_t>(vwn,
                                                              src_term.str());
          break;
        case LIBDNN_ACCUMULATE_PREC_16:
          ss << this->program_->template convert_type<int16_t>(vwn,
                                                               src_term.str());
          break;
        case LIBDNN_ACCUMULATE_PREC_32:
          ss << this->program_->template convert_type<int32_t>(vwn,
                                                               src_term.str());
          break;
        case LIBDNN_ACCUMULATE_PREC_64:
          ss << this->program_->template convert_type<int64_t>(vwn,
                                                               src_term.str());
          break;
        case LIBDNN_ACCUMULATE_PREC_NATIVE:
        default:
          ss << src_term.str() << std::endl;
          break;
      }
    }
  }
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Loop over a single tile
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNN<MItype, MOtype>::generate_accreg_init(
    shared_ptr<LibDNNTuner> tuner, bool dterm, bool load, bool beta_term,
    libdnnAccumulatePrecision_t prec) {
  stringstream ss;

  int vwm = tuner->get_param<int>("VWM");
  int vwn = tuner->get_param<int>("VWN");
  bool unroll = tuner->get_param<bool>("vector_unroll");

  string accreg_type = "MItype";
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
      ss << "((" << accreg_type << "*)(&(Dreg[wm / VWM])))[wm %V WM] = ";
      if (prec != LIBDNN_ACCUMULATE_PREC_NATIVE) {
        ss << "(" << accreg_type << ")";
      }
      ss << "(Dptr[globalRow]);" << std::endl;
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
    ss << "((" << accreg_type << "*)(&(Creg[wm][wn / VWN])))[wn % VWN] = ";
    if (prec != LIBDNN_ACCUMULATE_PREC_NATIVE) {
      ss << "(" << accreg_type << ")";
    }
    if (beta_term) {
      ss << "(beta * ";
    } else {
      ss << "(";
    }
    ss << "Cptr[globalRow * N + globalCol]);" << std::endl;
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
          ss << "VEC_" << vwm << "_" << i << "(Dreg[wm]) = ("
             << accreg_type << ")0;" << std::endl;
        }
      } else {
        ss << "Dreg[wm] = (" << accreg_type << ")0;" << std::endl;
      }
      ss << "}" << std::endl;
    }
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wm = 0; wm < WPTM; ++wm) {" << std::endl;
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {" << std::endl;
    if (unroll) {
      for (int i = 0; i < vwn; ++i) {
        ss << "VEC_" << vwn << "_" << i << "(Creg[wm][wn]) = ("
           << accreg_type << ")0;" << std::endl;
      }
    } else {
      ss << "Creg[wm][wn] = (" << accreg_type << ")0;" << std::endl;
    }
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }
  return ss.str();
}

INSTANTIATE_CLASS_2T_GUARDED(LibDNN, PROTO_TYPES, PROTO_TYPES);

}  // namespace caffe

#endif  // USE_LIBDNN
