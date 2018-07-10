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
std::map<string, int64_t> LibDNN<MItype, MOtype>
    ::gemm_like_default_parameters() {

  std::map<string, int64_t> params;

  if (this->dev_ptr_->name().find("Mali") != string::npos) {
    params["workgroup_size_0"] = 8;
    params["workgroup_size_1"] = 8;
    params["workgroup_size_2"] = 1;
    params["TSK"] = 8 / safe_sizeof<MItype>();;
    params["TSK_UNROLL"] = 8 / safe_sizeof<MItype>();
    params["WPTM"] = 8 / safe_sizeof<MItype>();
    params["WPTN"] = 8 / safe_sizeof<MItype>();
    params["VWM"] = 8 / safe_sizeof<MItype>();
    params["VWN"] = 8 / safe_sizeof<MItype>();
  }

  if (this->dev_ptr_->name().find("VideoCore IV") != string::npos) {
    params["workgroup_size_0"] = 4;
    params["workgroup_size_1"] = 3;
    params["workgroup_size_2"] = 1;
    params["TSK"] = 3;
    params["TSK_UNROLL"] = 1;
    params["WPTM"] = 4;
    params["WPTN"] = 4;
  }

  return params;
}

template<typename MItype, typename MOtype>
string LibDNN<MItype, MOtype>::generate_gemm_core(
    shared_ptr<LibDNNTuner> tuner, bool dterm, bool alpha_term,
    bool alpha_exactly_one) {
  stringstream ss;
  int wptm = tuner->get_param<int>("WPTM");
  int wptn = tuner->get_param<int>("WPTN");
  int vwm = tuner->get_param<int>("VWM");
  int vwn = tuner->get_param<int>("VWN");
  int rtsn = tuner->get_param<int>("workgroup_size_0");
  int rtsm = tuner->get_param<int>("workgroup_size_1");
  bool unroll = tuner->get_param<bool>("vector_unroll");
  int tsk_unroll = tuner->get_param<int>("TSK_UNROLL");
  bool dp4a = tuner->get_param<bool>("DP4A");
  bool no_reg_arrs = tuner->get_param<bool>("no_reg_arrs");


  // Temporary registers for A and B
  ss << "MItype" << vwm << " Areg;" << std::endl;
  if (no_reg_arrs) {
    for (int breg = 0; breg < tsk_unroll * wptn / vwn; ++breg) {
      ss << "MItype" << vwn << " Breg_" << breg << ";" << std::endl;
    }
  } else {
    ss << "MItype" << vwn << " Breg[TSK_UNROLL*WPTN/VWN];" << std::endl;
  }

  // Loop over the values of a single tile
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int_tp k = 0; k < TSK; k += TSK_UNROLL) {" << std::endl;

  if (no_reg_arrs) {
    ss << "int_tp col;" << std::endl;
    for (int_tp wn = 0; wn < wptn / (vwn / tsk_unroll); ++wn) {
      ss << "col = tidn + " << wn * (vwn / tsk_unroll) * rtsn << ";"
         << std::endl;
      for (int i = 0; i < vwn; ++i) {
        ss << "VEC_" << vwn << "_" << i << "(Breg_" << wn << ")"
           << " = Bsub[k + " << (i % tsk_unroll) << "][col + "
           << (i / tsk_unroll * rtsn) << "];" << std::endl;
      }
      if (is_integer_type<MItype>()) {
        ss << "if (tidm == 0) {" << std::endl;
        for (int i = 0; i < vwn; ++i) {
          ss << "Bsubcols[col + " << (i / tsk_unroll * rtsn) << "] += VEC_"
             << vwn << "_" << i << "(Breg_" << wn << ");" << std::endl;
        }
        ss << "}" << std::endl;
      }
    }
  } else {
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wn = 0; wn < WPTN / (VWN / TSK_UNROLL); ++wn) {"
       << std::endl;
    ss << "int_tp col = tidn + wn * (VWN / TSK_UNROLL) * RTSN;" << std::endl;
    for (int i = 0; i < vwn; ++i) {
      ss << "VEC_" << vwn << "_" << i << "(Breg[wn])"
         << " = Bsub[k + " << (i % tsk_unroll) << "][col + "
         << (i / tsk_unroll * rtsn) << "];" << std::endl;
    }
    if (is_integer_type<MItype>()) {
      ss << "if (tidm == 0) {" << std::endl;
      for (int i = 0; i < vwn; ++i) {
        ss << "Bsubcols[col + " << (i / tsk_unroll * rtsn) << "] += VEC_"
           << vwn << "_" << i << "(Breg[wn]);" << std::endl;
      }
      ss << "}" << std::endl;
    }
    ss << "}" << std::endl;
  }

  if (no_reg_arrs) {
    ss << "int_tp row;" << std::endl;
    for (int_tp wm = 0; wm < wptm / (vwm / tsk_unroll); ++wm) {
      // Perform the computation
      ss << "row = tidm + " << (wm * (vwm / tsk_unroll) * rtsm) << ";"
         << std::endl;
      for (int i = 0; i < vwm; ++i) {
        ss << "VEC_" << vwm << "_" << i << "(Areg)" << " = Asub[row + "
           << (i / tsk_unroll * rtsm)
           << "][k + " << (i % tsk_unroll) << "];" << std::endl;
      }
      if (is_integer_type<MItype>()) {
        ss << "if (tidn == 0) {" << std::endl;
        for (int i = 0; i < vwm; ++i) {
          ss << "Asubrows[row + " << (i / tsk_unroll * rtsm) << "] += VEC_"
             << vwm << "_" << i << "(Areg);" << std::endl;
        }
        ss << "}" << std::endl;
      }
      if (dterm) {
        if (unroll || tsk_unroll > 1) {
          for (int i = 0; i < vwm; ++i) {
            ss << "VEC_" << vwm << "_" << (i / tsk_unroll) << "(Dreg_"
               << wm << ") "
               << "+= " << "(Acctype)(VEC_" << vwm << "_" << i
               << "(Areg) * bias_mult);"
               << std::endl;
          }
        } else {
          ss << "Dreg_" << wm << " += ";
          ss << "Areg * (MItype)bias_mult";
          ss << ";" << std::endl;
        }
      }
      for (int_tp wn = 0; wn < wptn / vwn; ++wn) {
        if (unroll || tsk_unroll > 1) {
          for (int_tp n = 0; n < vwn; ++n) {
            for (int_tp m = 0; m < vwm / tsk_unroll; ++m) {
              ss << "VEC_" << vwn << "_" << n
                 << "(Creg_" << (wm  * vwm / tsk_unroll + m) << "_" << wn << ")"
                 << " += ";
              ss << "(Acctype)";
              if (alpha_term && !alpha_exactly_one && is_float_type<MItype>()) {
                ss << "(alpha * ";
              } else {
                ss << "(";
              }
              if (dp4a && tsk_unroll == 4 && vwn == 4 && vwm == 4 && false) {
                ss << "__dp4a(Areg, Breg_" << (wn * tsk_unroll +
                                               (n / (vwn / tsk_unroll)))
                   << ", 0));" << std::endl;
              } else {
                ss << "(";
                for (int_tp ku = 0; ku < tsk_unroll; ++ku) {
                  ss << "VEC_" << vwm << "_" << (m * tsk_unroll + ku)
                     << "(Areg) * VEC_" << vwn << "_"
                     << ((n % (vwn / tsk_unroll)) * tsk_unroll + ku)
                     << "(Breg_" << (wn * tsk_unroll +
                                     (n / (vwn / tsk_unroll))) << ")";
                  if (ku < tsk_unroll - 1) {
                    ss << " + ";
                  } else {
                    ss << "));" << std::endl;
                  }
                }
              }
            }
          }
        } else {
          for (int_tp m = 0; m < vwm; ++m) {
            ss << "Creg_" << wm  * vwm / tsk_unroll + m << "_" << wn << " += ";
            stringstream src_term;
            if (alpha_term && !alpha_exactly_one && is_float_type<MItype>()) {
              src_term << "(alpha *";
            } else {
              src_term << "(";
            }
            src_term << "VEC_" << vwm << "_" << m << "(Areg)"
                     << " * (Breg_" << wn << "))" << std::endl;
          }
        }
      }
    }
  } else {
    // Perform the computation
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wm = 0; wm < WPTM / (VWM / TSK_UNROLL); ++wm) {"
       << std::endl;
    ss << "int_tp row = tidm + wm * (VWM / TSK_UNROLL) * RTSM;" << std::endl;
    for (int i = 0; i < vwm; ++i) {
      ss << "VEC_" << vwm << "_" << i << "(Areg)" << " = Asub[row + "
         << (i / tsk_unroll * rtsm)
         << "][k + " << (i % tsk_unroll) << "];" << std::endl;
    }
    if (is_integer_type<MItype>()) {
      ss << "if (tidn == 0) {" << std::endl;
      for (int i = 0; i < vwm; ++i) {
        ss << "Asubrows[row + " << (i / tsk_unroll * rtsm) << "] += VEC_"
           << vwm << "_" << i << "(Areg);" << std::endl;
      }
      ss << "}" << std::endl;
    }
    if (dterm) {
      if (unroll || tsk_unroll > 1) {
        for (int i = 0; i < vwm; ++i) {
          ss << "VEC_" << vwm << "_" << (i / tsk_unroll) << "(Dreg[wm]) "
             << "+= " << "(Acctype)(VEC_" << vwm << "_" << i
             << "(Areg) * bias_mult);"
             << std::endl;
        }
      } else {
        ss << "Dreg[wm] += ";
        ss << "Areg * (MItype)bias_mult";
        ss << ";" << std::endl;
      }
    }
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {"
       << std::endl;
    if (unroll || tsk_unroll > 1) {
      for (int_tp n = 0; n < vwn; ++n) {
        for (int_tp m = 0; m < vwm / tsk_unroll; ++m) {
          ss << "VEC_" << vwn << "_" << n << "(Creg[wm * VWM / TSK_UNROLL + "
             << m << "][wn])" << " += ";
          ss << "(Acctype)";
          if (alpha_term && !alpha_exactly_one && is_float_type<MItype>()) {
            ss << "(alpha * ";
          } else {
            ss << "(";
          }
          if (dp4a && tsk_unroll == 4 && vwn == 4 && vwm == 4 && false) {
            ss << "__dp4a(Areg, Breg[wn * TSK_UNROLL + "
               << (n / (vwn / tsk_unroll)) << "], 0));" << std::endl;
          } else {
            ss << "(";
            for (int_tp ku = 0; ku < tsk_unroll; ++ku) {
              ss << "VEC_" << vwm << "_" << (m * tsk_unroll + ku) << "(Areg)"
                 << " * VEC_" << vwn << "_"
                 << ((n % (vwn / tsk_unroll)) * tsk_unroll + ku)
                 << "(Breg[wn * TSK_UNROLL + "
                 << (n / (vwn / tsk_unroll)) << "])";
              if (ku < tsk_unroll - 1) {
                ss << " + ";
              } else {
                ss << "));" << std::endl;
              }
            }
          }
        }
      }
    } else {
      for (int_tp m = 0; m < vwm; ++m) {
        ss << "Creg[wm * VWM + " << m << "][wn] += ";
        stringstream src_term;
        if (alpha_term && !alpha_exactly_one && is_float_type<MItype>()) {
          src_term << "(alpha *";
        } else {
          src_term << "(";
        }
        src_term << "VEC_" << vwm << "_" << m << "(Areg)"
                 << " * (Breg[wn]))" << std::endl;
      }
    }
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  // Loop over a single tile
  ss << "}" << std::endl;

  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNN<MItype, MOtype>::generate_accreg_init(
    shared_ptr<LibDNNTuner> tuner, bool dterm, bool load, bool beta_term,
    bool beta_exactly_one) {
  stringstream ss;
  int wptm = tuner->get_param<int>("WPTM");
  int wptn = tuner->get_param<int>("WPTN");
  int vwm = tuner->get_param<int>("VWM");
  int vwn = tuner->get_param<int>("VWN");
  int rtsn = tuner->get_param<int>("workgroup_size_0");
  int rtsm = tuner->get_param<int>("workgroup_size_1");
  bool unroll = tuner->get_param<bool>("vector_unroll");
  bool no_reg_arrs = tuner->get_param<bool>("no_reg_arrs");

  if (no_reg_arrs) {
    if (dterm) {
      for (int_tp wm = 0; wm < wptm / vwm; ++wm) {
        ss << "Acctype" << vwm << " Dreg_" << wm << ";" << std::endl;
      }
    }
    for (int_tp wm = 0; wm < wptm; ++wm) {
      for (int_tp wn = 0; wn < wptn / vwn; ++wn) {
        ss << "Acctype" << vwn << " Creg_" << wm << "_" << wn << ";"
           << std::endl;
      }
    }
  } else {
    if (dterm) {
      ss << "Acctype" << vwm << " Dreg[WPTM / VWM];" << std::endl;
    }
    ss << "Acctype" << vwn << " Creg[WPTM][WPTN / VWN];" << std::endl;
  }

  // Initialize the accumulation registers (only preload with float types)
  // Quantized types require adding the values post-GEMM due to offsets
  if (dterm) {
    if (no_reg_arrs) {
      ss << "{" << std::endl;
      ss << "int_tp globalRow;" << std::endl;
      for (int_tp wm = 0; wm < wptm; ++wm) {
        if (load && is_float_type<MOtype>()) {
          ss << "globalRow = offM + tidm + " << (wm * rtsm) << ";" << std::endl;
          ss << "if (globalRow < M) {" << std::endl;
        }
        ss << "((Acctype*)(&(Dreg_" << (wm / vwm) << ")))[" << (wm % vwm)
           << "] = ";
        if (load && is_float_type<MOtype>()) {
          ss << "(Acctype)(Dptr[globalRow]);" << std::endl;
          ss << "}" << std::endl;
        } else {
          ss << "(Acctype)0;" << std::endl;
        }
      }
      ss << "}" << std::endl;
    } else {
      ss << "#pragma unroll" << std::endl;
      ss << "for (int_tp wm = 0; wm < WPTM; ++wm) {" << std::endl;
      if (load && is_float_type<MOtype>()) {
        ss << "int_tp globalRow = offM + tidm + wm * RTSM;" << std::endl;
        ss << "if (globalRow < M) {" << std::endl;
      }
      ss << "((Acctype*)(&(Dreg[wm / VWM])))[wm % VWM] = ";
      if (load && is_float_type<MOtype>()) {
        ss << "(Acctype)(Dptr[globalRow]);" << std::endl;
        ss << "}" << std::endl;
      } else {
        ss << "(Acctype)0;" << std::endl;
      }
      ss << "}" << std::endl;
    }
  }
  stringstream ss_beta_c;
  if (beta_term && load && is_float_type<MOtype>()) {
    if (beta_exactly_one) {
      ss_beta_c << "Cptr[globalRow * N + globalCol];" << std::endl;
    } else {
      // Float code
      ss_beta_c << "(beta * Cptr[globalRow * N + globalCol]);"
                << std::endl;
    }
  } else {
    ss_beta_c << "0;" << std::endl;
  }
  if (no_reg_arrs) {
    ss << "{" << std::endl;
    ss << "int_tp globalRow;" << std::endl;
    ss << "int_tp globalCol;" << std::endl;
    for (int_tp wm = 0; wm < wptm; ++wm) {
      for (int_tp wn = 0; wn < wptn; ++wn) {
        if (load && is_float_type<MOtype>()) {
          ss << "globalRow = offM + tidm + " << (wm * rtsm) << ";"
             << std::endl;
          ss << "globalCol = offN + tidn + " << (wn * rtsn) << ";"
             << std::endl;
          ss << "if (globalRow < M && globalCol < N) {" << std::endl;
        }
        ss << "((Acctype*)(&(Creg_" << wm << "_" << (wn / vwn) << ")))["
           << (wn % vwn) << "] = " << "(Acctype)" << ss_beta_c.str();
        if (load && is_float_type<MOtype>()) {
          ss << "}" << std::endl;
        }
      }
    }
    ss << "}" << std::endl;
  } else {
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wm = 0; wm < WPTM; ++wm) {" << std::endl;
    if (load && is_float_type<MOtype>()) {
      ss << "int_tp globalRow = offM + tidm + wm * RTSM;" << std::endl;
    }
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wn = 0; wn < WPTN; ++wn) {" << std::endl;
    if (load && is_float_type<MOtype>()) {
      ss << "int_tp globalCol = offN + tidn + wn * RTSN;" << std::endl;
    }
    if (load && is_float_type<MOtype>()) {
      ss << "if (globalRow < M && globalCol < N) {" << std::endl;
    }
    ss << "((Acctype*)(&(Creg[wm][wn / VWN])))[wn % VWN] = ";
    ss << "(Acctype)" << ss_beta_c.str();
    if (load && is_float_type<MOtype>()) {
      ss << "}" << std::endl;
    }
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }
  return ss.str();
}

INSTANTIATE_CLASS_2T_GUARDED(LibDNN, PROTO_TYPES, PROTO_TYPES);

}  // namespace caffe

#endif  // USE_LIBDNN
