#ifdef USE_LIBDNN

#include "caffe/libdnn/libdnn_blas.hpp"

namespace caffe {

template<typename MItype, typename MOtype>
void LibDNNBlas<MItype, MOtype>::initialize_gemv_tuner(
    shared_ptr<DeviceProgram> program, shared_ptr<LibDNNTuner> tuner) {
  // Work groups
  for (int id = 0; id < 1; ++id) {
    vector<int_tp> workgroup_sizes;
    for (int_tp i = 0; i < this->dev_ptr_->workgroup_size(id);
            i += 4) {
      workgroup_sizes.push_back(i);
    }
    tuner->add_set_param <int_tp>("workgroup_size_" + std::to_string(id),
                                  64, workgroup_sizes);
  }

  tuner->add_set_param<int_tp>("WPTM", 2, vector<int_tp>({1, 2, 4, 8, 16 }));
  tuner->add_set_param<int_tp>("WPTN", 2, vector<int_tp>({1, 2, 4, 8, 16 }));
  tuner->add_set_param<int_tp>("VWM", 2, vector<int_tp>({1, 2, 4, 8, 16 }));
  tuner->add_set_param<int_tp>("VWN", 2, vector<int_tp>({1, 2, 4, 8, 16 }));

  tuner->add_constraint<int64_t>(
    vector<string>({"WPTM", "VWM"}),
    vector<string>({"WPTM"}), [](vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });

  tuner->add_constraint<int64_t>(
    vector<string>({"WPTN", "VWN"}),
    vector<string>({"WPTN"}), [](vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });

  if (this->dev_ptr_->backend() == BACKEND_CUDA) {
    // CUDA needs the vector elements unrolled
    tuner->add_boolean_param("vector_unroll", true, false);
  } else {
    // OpenCL does not need the vector elements unrolled, and may
    // save registers by not doing it
    tuner->add_boolean_param("vector_unroll", true, true);
  }
}

template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::generate_gemv_source(
    shared_ptr<DeviceProgram> program, shared_ptr<LibDNNTuner> tuner,
    bool trans_A, const uint_tp M, const uint_tp N,
    bool alpha_term, bool alpha_exactly_one,
    bool beta_term, bool beta_exactly_one) {

  typedef typename std::conditional<float_is_same<MItype>::value, MItype,
          typename std::conditional<sizeof(MItype) == 1, int16_t,
          typename std::conditional<sizeof(MItype) == 2, int32_t,
                                    int64_t>::type>::type>::type Difftype;
  typedef typename std::conditional<float_is_same<MItype>::value, MItype,
          typename std::conditional<sizeof(MItype) == 1, int32_t,
                                    int64_t>::type>::type Acctype;

  stringstream ss;

  ss << program->setup();
  ss << program->template define_vector_type<MItype>("MItype", 0, 16);
  ss << program->template define_vector_type<MOtype>("MOtype", 0, 16);
  ss << program->template define_vector_type<Acctype>("Acctype", 0, 16);
  ss << program->template define_vector_type<Difftype>("Difftype", 0, 16);
  if (is_integer_type<MItype>()) {
    if (this->dev_ptr_->template preferred_vector_width<int64_t>() > 0) {
      ss << program->template define_vector_type<int64_t>("Multtype", 0, 16);
    } else {
      ss << program->template define_vector_type<int32_t>("Multtype", 0, 16);
    }
  }
  ss << program->vector_accessors();

  int wptm = tuner->get_param<int>("WPTM");
  int wptn = tuner->get_param<int>("WPTN");
  int vwm = tuner->get_param<int>("VWM");
  int vwn = tuner->get_param<int>("VWN");
  int rtsn = tuner->get_param<int>("workgroup_size_0");
  int tsn = wptn * rtsn;
  bool unroll = tuner->get_param<bool>("vector_unroll");

  // Split up calculation if vector x is not divisible by rtsn
  bool needs_split = (N % tsn != 0);
  // Check if the big block computation is needed at all
  bool needs_block = (N >= tsn);

  // GEMV definitions
  ss << program->define("M", M);
  ss << program->define("N", N);

  // The work-per-thread in dimension M
  ss << program->define("WPTM", tuner->get_param<int>("WPTM"));
  ss << program->define("VWM", tuner->get_param<int>("VWM"));
  // The work-per-thread in dimension N
  ss << program->define("WPTN", tuner->get_param<int>("WPTN"));
  ss << program->define("VWN", tuner->get_param<int>("VWN"));
  // The reduced tile-size in dimension N
  ss << program->define("RTSN", tuner->get_param<int>("workgroup_size_0"));
  ss << program->define("TSN", tuner->get_param<int>("WPTN")
          * tuner->get_param<int>("workgroup_size_0"));


  KernelArgs args;
  if (is_integer_type<MItype>() || is_integer_type<MOtype>()) {
    args.push_back(program->template create_kernel_arg<int8_t>("shift_bits",
                                                             KERNEL_ARG_CONST));
  }
  if (alpha_term && !alpha_exactly_one) {
    args.push_back(program->template create_kernel_arg<MOtype>("alpha",
                                                             KERNEL_ARG_CONST));
    if (is_integer_type<MOtype>()) {
      args.push_back(program->template create_kernel_arg<MOtype>("alpha_off",
                                                             KERNEL_ARG_CONST));
      args.push_back(program->template create_kernel_arg<int32_t>("alpha_mult",
                                                             KERNEL_ARG_CONST));
      args.push_back(program->template create_kernel_arg<int8_t>("alpha_shift",
                                                             KERNEL_ARG_CONST));
    }
  }
  args.push_back(program->template create_kernel_arg<MItype>("A",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  if (is_integer_type<MItype>()) {
    args.push_back(program->template create_kernel_arg<MItype>("A_off",
                                                             KERNEL_ARG_CONST));
  }
  args.push_back(program->template create_kernel_arg<MItype>("x",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  if (is_integer_type<MItype>()) {
    args.push_back(program->template create_kernel_arg<MItype>("x_off",
                                                             KERNEL_ARG_CONST));
  }
  if (beta_term && !beta_exactly_one) {
    args.push_back(program->template create_kernel_arg<MOtype>("beta",
                                                             KERNEL_ARG_CONST));
    if (is_integer_type<MOtype>()) {
      args.push_back(program->template create_kernel_arg<MOtype>("beta_off",
                                                             KERNEL_ARG_CONST));
      args.push_back(program->template create_kernel_arg<int32_t>("beta_mult",
                                                             KERNEL_ARG_CONST));
      args.push_back(program->template create_kernel_arg<int8_t>("beta_shift",
                                                             KERNEL_ARG_CONST));
    }
  }
  args.push_back(program->template create_kernel_arg<MOtype>("y",
                                  KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  if (is_integer_type<MOtype>()) {
    args.push_back(program->template create_kernel_arg<MOtype>("y_off",
                                                             KERNEL_ARG_CONST));
    args.push_back(program->template create_kernel_arg<Acctype>("y_min",
                                                             KERNEL_ARG_CONST));
    args.push_back(program->template create_kernel_arg<Acctype>("y_max",
                                                             KERNEL_ARG_CONST));
    args.push_back(program->template create_kernel_arg<int32_t>("mult",
                                                             KERNEL_ARG_CONST));
    args.push_back(program->template create_kernel_arg<int8_t>("shift",
                                                             KERNEL_ARG_CONST));
  }

  ss << program->function("libdnn_gemv", args);

  // Thread identifiers
  // Local row ID (max: RTSM=TSM/WPTM)
  ss << "const int_tp tidn = " << program->local_id(0) << ";" << std::endl;
  // Work-group offset
  ss << "const int_tp offM = WPTM * " << program->global_id(0) << ";"
     << std::endl;

  // Local memory
  // xsub
  ss << program->local_mem("Difftype", "xsub[" + std::to_string(tsn) + "]")
     << ";" << std::endl;

  ss << "Acctype" << vwm << " yreg[WPTM / VWM];" << std::endl;

  // Initialize accumulator to 0
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm = 0; wm < WPTM; ++wm) {" << std::endl;
  if (unroll) {
    for (int m = 0; m < vwm; ++m) {
      ss << "VEC_" << vwm << "_" << m << "(yreg[wm]) = (Acctype)0;"
         << std::endl;
    }
  } else {
    ss << "yreg[wm] = (Acctype)0;" << std::endl;
  }
  ss << "}" << std::endl;

  if (alpha_term) {
    for(int_tp mode = 0; mode < 2; ++mode) {
      if ((needs_block && mode == 0) || (needs_split && mode == 1)) {
        if (mode == 0) {
          ss << "for (int_tp offN = 0; offN < N - (N % TSN); offN += TSN) {"
             << std::endl;
        } else {
          ss << "int_tp offN = N - (N % TSN);" << std::endl;
        }
        // Every thread loads WPTN elements of x into xsub (local memory)
        for (int_tp n = 0; n < wptn; ++n) {
          if (mode == 1) {
            ss << "if (tidn * WPTN + offN + " << n << " < N) {" << std::endl;
          }
          ss << "xsub[tidn * WPTN + " << n << "] = "
             << "(Difftype)(x[tidn * WPTN + offN + " << n << "])";
          if (is_float_type<MItype>()) {
            ss << ";" << std::endl;
          } else {
            ss << "- (Difftype)(x_off);" << std::endl;
          }
          if (mode == 1) {
            ss << "} else {" << std::endl;
            ss << "xsub[tidn * WPTN + " << n << "] = "
               << "(Difftype)0;" << std::endl;
            ss << "}" << std::endl;
          }
        }
        // Barrier after loading xsub
        ss << program->local_barrier() << std::endl;
        // Temporary registers for A and x
        ss << "Difftype" << vwm << " Areg[WPTM][WPTN / VWN];" << std::endl;
        ss << "Difftype" << vwn << " xreg[WPTN / VWN];" << std::endl;

        // Loop over work
        ss << "for (int_tp t = 0; t < RTSN; ++t) {" << std::endl;
        // Load x into registers
        ss << "for (int_tp wn = 0; wn < WPTN/VWN; ++wn) {" << std::endl;
        if (unroll) {
          for (int_tp n = 0; n < vwn; ++n) {
            ss << "VEC_" << vwn << "_" << n << "(xreg[wn]) = "
               << "xsub[wn + " << n << " + t * WPTN];" << std::endl;
          }
        } else {
          ss << "xreg[wn] = xsub[wn + t * WPTN];" << std::endl;
        }
        ss << "}" << std::endl;  // WPTN/VWN
        // Load A into registers
        ss << "for (int_tp wm = 0; wm < WPTM; ++wm) {" << std::endl;
        ss << "if (wm + offM < M) {" << std::endl;
        ss << "for (int_tp wn = 0; wn < WPTN/VWN; ++wn) {" << std::endl;

        for (int n = 0; n < vwn; ++n) {
          if (mode == 1) {
            ss << "if (wn + t * WPTN + offN + " << n << " < N) {" << std::endl;
          }
          if (trans_A) {
            ss << "VEC_" << vwn << "_" << n << "(Areg[wm][wn])"
               << " = (Difftype)(A[(wn + t * WPTN + offN + " << n
               << ") * M + wm + offM])";
          } else {
            ss << "VEC_" << vwn << "_" << n << "(Areg[wm][wn])"
               << " = (Difftype)(A[wn + t * WPTN + offN + " << n
               << " + (wm + offM) * N])";
          }
          if (is_float_type<MItype>()) {
            ss << ";" << std::endl;
          } else {
            ss << "- (Difftype)(A_off);" << std::endl;
          }
          if (mode == 1) {
            ss << "} else {" << std::endl;
            ss << "VEC_" << vwn << "_" << n << "(Areg[wm][wn]) = (Difftype)0;"
               << std::endl;
            ss << "}" << std::endl;
          }
        }
        ss << "}" << std::endl;  // WPTN/VWN
        ss << "} else {" << std::endl;  // M-Guard
        ss << "for (int_tp wn = 0; wn < WPTN/VWN; ++wn) {" << std::endl;
        if (unroll) {
          for (int n = 0; n < vwn; ++n) {
            ss << "VEC_" << vwn << "_" << n << "(Areg[wm][wn]) = (Difftype)0;"
               << std::endl;
          }
        } else {
          ss << "Areg[wm][wn] = (Difftype)0;" << std::endl;
        }
        ss << "}" << std::endl;  // WPTN/VWN
        ss << "}" << std::endl;  // M-Guard
        ss << "}" << std::endl;  // WPTM
        // Compute
        ss << "#pragma unroll" << std::endl;
        ss << "for (int_tp wm = 0; wm < WPTM / VWM; ++wm) {" << std::endl;
        ss << "#pragma unroll" << std::endl;
        ss << "for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {" << std::endl;
        if (unroll) {
          for (int_tp m = 0; m < vwm; ++m) {
            for (int_tp n = 0; n < vwn; ++n) {
              ss << "VEC_" << vwm << "_" << m << "(yreg[wm])"
                 << " += ";
              ss << "(Acctype)" << "(VEC_" << vwn << "_" << n
                 << "(Areg[wm * VWM + " << m << "][wn]))"
                 << " * (Acctype)(VEC_" << vwn << "_" << n << "(xreg[wn]));"
                 << std::endl;
            }
          }
        } else {
          for (int_tp m = 0; m < vwm; ++m) {
            ss << "Acctype" << vwn << " tmp;" << std::endl;
            ss << "tmp = (Acctype)(Areg[wm * VWM + " << m
               << "][wn]) * (Acctype)(xreg[wn]);" << std::endl;
            for (int_tp n = 0; n < vwn; ++n) {
              ss << "VEC_" << vwm << "_" << m << "(yreg[wm])"
                 << " += VEC_" << vwn << "_" << n << "(tmp)";
            }
          }
        }
        ss << "}" << std::endl;  // wn
        ss << "}" << std::endl;  // wm
        ss << "}" << std::endl;  // t < RTSN
        // Barrier after using all of xsub
        ss << program->local_barrier() << std::endl;
        if (mode == 0) {
          ss << "}" << std::endl;  // N - (N % TSN)
        }
      }
    }
  }  // if (alpha_term)

  // Store y
  ss << "for (int_tp wm = 0; wm < WPTM/VWM; ++wm) {" << std::endl;
  for (int_tp m = 0; m < vwm; ++m) {
    ss << "if (wm + offM + " << m << " < M) {" << std::endl;

    if (is_float_type<MItype>()) {
      // Float type code
      ss << "y[offM + wm + " << m << "] = ";
      if (alpha_term) {
        if (!alpha_exactly_one) {
          ss << "alpha * ";
        }
        ss << "VEC_" << vwm << "_" << m << "(yreg[wm])";
      }
      if (alpha_term && beta_term) {
        ss << " + ";
      }
      if (beta_term) {
        if (!beta_exactly_one) {
          ss << "beta * ";
        }
        ss << "y[offM + wm + " << m << "]";
      }
      if (!alpha_term && !beta_term) {
        ss << "(Acctype)0.0;" << std::endl;
      }
      ss << ";" << std::endl;
    } else {
      // Quantization type code
      if (alpha_term) {
        ss << "{" << std::endl;
        ss << "Acctype y_tmp = (Acctype)(((VEC_" << vwm << "_" << m
           << "(yreg[wm])) * ((Multtype)mult))"
           << "/ ((Multtype)1 << shift_bits));" << std::endl;
        ss << "if (shift >= 0) {" << std::endl;
        ss << "Acctype mask = ((Acctype)1 << shift) - (Acctype)1;"
           << std::endl;
        ss << "Acctype y_round = (y_tmp & mask) > "
           << "((mask >> 1) + ((y_tmp < (Acctype)0) ? "
           << "(Acctype)1 : (Acctype)0)) ? "
           << "(Acctype)1 : (Acctype)0;" << std::endl;
        ss << "y_tmp = (y_tmp >> shift) + y_round;" << std::endl;
        ss << "} else {" << std::endl;
        ss << "y_tmp = y_tmp << -shift;" << std::endl;
        ss << "}" << std::endl;
        if (!alpha_exactly_one) {
          ss << "Difftype alpha_diff = alpha - alpha_off;" << std::endl;
          ss << "y_tmp = ((Acctype)alpha_diff) * ((Acctype)y_tmp);"
             << std::endl;
          ss << "y_tmp = (Acctype)((((Multtype)y_tmp) * ((Multtype)alpha_mult))"
             << "/ ((Multtype)1 << shift_bits));" << std::endl;
          ss << "if (alpha_shift >= 0) {" << std::endl;
          ss << "Acctype mask = ((Acctype)1 << alpha_shift) - (Acctype)1;"
             << std::endl;
          ss << "Acctype y_round = (y_tmp & mask) > "
             << "((mask >> 1) + ((y_tmp < (Acctype)0) ? "
             << "(Acctype)1 : (Acctype)0)) ? "
             << "(Acctype)1 : (Acctype)0;" << std::endl;
          ss << "y_tmp = (y_tmp >> alpha_shift) + y_round;" << std::endl;
          ss << "} else {" << std::endl;
          ss << "y_tmp = y_tmp << -alpha_shift;" << std::endl;
          ss << "}" << std::endl;
        }
        ss << "VEC_" << vwm << "_" << m << "(yreg[wm]) = y_tmp;" << std::endl;
        ss << "}" << std::endl;
      }
      if (beta_term) {
        ss << "{" << std::endl;
        ss << "Acctype y_tmp = (Acctype)(y[offM + wm + " << m << "])"
           << " - ((Acctype)y_off);" << std::endl;
        if (!beta_exactly_one) {
          ss << "Difftype beta_diff = beta - beta_off;" << std::endl;
          ss << "y_tmp = ((Acctype)beta_diff) * ((Acctype)y_tmp);"
             << std::endl;
          ss << "y_tmp = (Acctype)((((Multtype)y_tmp) * ((Multtype)beta_mult))"
             << "/ ((Multtype)1 << shift_bits));" << std::endl;
          ss << "if (beta_shift >= 0) {" << std::endl;
          ss << "Acctype mask = ((Acctype)1 << beta_shift) - (Acctype)1;"
             << std::endl;
          ss << "Acctype y_round = (y_tmp & mask) > "
             << "((mask >> 1) + ((y_tmp < (Acctype)0) ? "
             << "(Acctype)1 : (Acctype)0)) ? "
             << "(Acctype)1 : (Acctype)0;" << std::endl;
          ss << "y_tmp = (y_tmp >> beta_shift) + y_round;" << std::endl;
          ss << "} else {" << std::endl;
          ss << "y_tmp = y_tmp << -beta_shift;" << std::endl;
          ss << "}" << std::endl;
        }
        ss << "VEC_" << vwm << "_" << m << "(yreg[wm]) += y_tmp;" << std::endl;
        ss << "}" << std::endl;
      }
      ss << "VEC_" << vwm << "_" << m << "(yreg[wm]) += y_off;" << std::endl;
      ss << "y[offM + wm + " << m << "] = (MOtype)"
         << "(min(max(VEC_" << vwm << "_" << m << "(yreg[wm]), y_min), y_max));"
         << std::endl;
    }
    ss << "}" << std::endl;  // M-Guard
  }
  ss << "}" << std::endl;
  // Kernel
  ss << "}" << std::endl;
  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::gemv_string_identifier(
    const CBLAS_TRANSPOSE trans_A, const uint_tp M, const uint_tp N,
    bool alpha_term, bool alpha_exactly_one,
    bool beta_term, bool beta_exactly_one) {
  stringstream ss;
  ss << "gemv_";
  ss << (trans_A == CblasNoTrans ? "TA_" : "NTA_");
  ss << "M" << M << "_";
  ss << "N" << N << "_";
  if (alpha_term) {
    ss << "alpha_";
    if (alpha_exactly_one) {
      ss << "1_";
    }
  }
  if (beta_term) {
    ss << "beta_";
    if (beta_exactly_one) {
      ss << "1_";
    }
  }
  return ss.str();
}


template<typename MItype, typename MOtype>
void LibDNNBlas<MItype, MOtype>::gemv(
               const CBLAS_TRANSPOSE trans_A,
               const uint_tp M, const uint_tp N,
               const MOtype alpha, vptr<const MItype> A, vptr<const MItype> x,
               const MOtype beta, vptr<MOtype> y,
               const QuantizerValues* const alpha_quant,
               const QuantizerValues* const a_quant,
               const QuantizerValues* const x_quant,
               const QuantizerValues* const beta_quant,
               const QuantizerValues* const y_quant) {

  typedef typename std::conditional<float_is_same<MItype>::value, MItype,
          typename std::conditional<sizeof(MItype) == 1, int16_t,
          typename std::conditional<sizeof(MItype) == 2, int32_t,
                                    int64_t>::type>::type>::type Difftype;
  typedef typename std::conditional<float_is_same<MItype>::value, MItype,
          typename std::conditional<sizeof(MItype) == 1, int32_t,
                                    int64_t>::type>::type Acctype;

  bool alpha_term = (alpha != MOtype(0)) || alpha_quant;
  bool beta_term = (beta != MOtype(0)) || beta_quant;
  bool alpha_exactly_one = (alpha == MOtype(1)) && !alpha_quant;
  bool beta_exactly_one = (beta == MOtype(1)) && !beta_quant;

  uint_tp MT = trans_A == CblasNoTrans ? M : N;
  uint_tp NT = trans_A == CblasNoTrans ? N : M;

  string identifier = gemv_string_identifier(trans_A, MT, NT,
                                             alpha_term, alpha_exactly_one,
                                             beta_term, beta_exactly_one);

  int_tp id = get_id(identifier);
  if (id < 0) {
    id = get_id_or_new(identifier);
  }
  shared_ptr<LibDNNTuner> tuner = program_tuners_[id];
  shared_ptr<DeviceProgram> program = programs_[id];
  boost::shared_lock<boost::shared_mutex> lock(program_mutex_);
  if (!program_ready_[id]) {
    lock.unlock();
    // Compiling new kernel has to lock the program lock exclusively
    boost::unique_lock<boost::shared_mutex> ulock(program_mutex_);
    if (!program_ready_[id]) {
      initialize_gemv_tuner(program, tuner);
      stringstream ss;
      ss << generate_gemv_source(program, tuner,
                                 trans_A == CblasTrans,
                                 MT, NT,
                                 alpha_term, alpha_exactly_one,
                                 beta_term, beta_exactly_one);
      program->set_source(ss.str());
      program->Compile(true, true);
      program_ready_[id] = true;
    }
    ulock.unlock();
    lock.lock();
  }
  lock.unlock();

  shared_ptr<DeviceKernel> kernel = program->GetKernel("libdnn_gemv");

  size_t wptm = tuner->get_param<int>("WPTM");
  size_t wgs0 = tuner->get_param<int>("workgroup_size_0");
  size_t div_M = wptm;

  vector<size_t> group = {((MT - 1) / div_M + 1), 1, 1};
  vector<size_t> local = {wgs0, 1, 1};

  // Quantization parameters
  int8_t shift_bits =
      (this->dev_ptr_->template preferred_vector_width<int64_t>() > 0 ? 32 : 16)
      / sizeof(MItype) - 1;

  MItype A_off;
  MItype x_off;
  MOtype y_off;
  int32_t mult;
  int8_t shift;
  Acctype y_min;
  Acctype y_max;
  MOtype alpha_off;
  int32_t alpha_mult;
  int8_t alpha_shift;
  MOtype beta_off;
  int32_t beta_mult;
  int8_t beta_shift;

  if (is_integer_type<MItype>() || is_integer_type<MOtype>()) {
    A_off = a_quant->get_zero<MItype>();
    x_off = x_quant->get_zero<MItype>();
    y_off = y_quant->get_zero<Acctype>();
    y_min = y_quant->get_min<Acctype>();
    y_max = y_quant->get_max<Acctype>();
    alpha_off = alpha_quant ? alpha_quant->get_zero<MOtype>() : MOtype(0);
    beta_off = beta_quant ? beta_quant->get_zero<MOtype>() : MOtype(0);

    QuantizerBase::template MultiplicativeQuantVals<int32_t>(
        a_quant, x_quant, y_quant, &mult, &shift, shift_bits);
    QuantizerBase::template MultiplicativeQuantVals<int32_t>(
        y_quant, alpha_quant, y_quant, &alpha_mult, &alpha_shift, shift_bits);
    QuantizerBase::template MultiplicativeQuantVals<int32_t>(
        y_quant, beta_quant, y_quant, &beta_mult, &beta_shift, shift_bits);
  }

  if (is_integer_type<MItype>() || is_integer_type<MOtype>()) {
    kernel->add_arg(&shift_bits);
  }
  if (alpha_term && !alpha_exactly_one) {
    kernel->add_arg(&alpha);
    if (is_integer_type<MOtype>()) {
      kernel->add_arg(&alpha_off);
      kernel->add_arg(&alpha_mult);
      kernel->add_arg(&alpha_shift);
    }
  }
  kernel->add_arg(&A);
  if (is_integer_type<MItype>()) {
    kernel->add_arg(&A_off);
  }
  kernel->add_arg(&x);
  if (is_integer_type<MItype>()) {
    kernel->add_arg(&x_off);
  }
  if (beta_term && !beta_exactly_one) {
    kernel->add_arg(&beta);
    if (is_integer_type<MOtype>()) {
      kernel->add_arg(&beta_off);
      kernel->add_arg(&beta_mult);
      kernel->add_arg(&beta_shift);
    }
  }
  kernel->add_arg(&y);
  if (is_integer_type<MOtype>()) {
    kernel->add_arg(&y_off);
    kernel->add_arg(&y_min);
    kernel->add_arg(&y_max);
    kernel->add_arg(&mult);
    kernel->add_arg(&shift);
  }
  kernel->Execute(group, local);
}

INSTANTIATE_CLASS_2T_GUARDED(LibDNNBlas, PROTO_TYPES, PROTO_TYPES);

}  // namespace caffe

#endif  // USE_LIBDNN
