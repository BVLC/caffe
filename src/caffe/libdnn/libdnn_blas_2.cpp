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
    bool alpha_term, bool beta_term,
    libdnnAccumulatePrecision_t prec,
    shared_ptr<Quantizer<MItype, MOtype> > top_quantize) {
  stringstream ss;

  ss << program->setup();
  ss << program->template define_vector_type<MItype>("MItype", 0, 16);
  ss << program->template define_vector_type<MOtype>("MOtype", 0, 16);
  ss << program->vector_accessors();

  string accreg_type = "MItype";
  switch (prec) {
    case LIBDNN_ACCUMULATE_PREC_NATIVE:
      break;
    case LIBDNN_ACCUMULATE_PREC_8:
      accreg_type = program->template device_type_name<int8_t>();
      break;
    case LIBDNN_ACCUMULATE_PREC_16:
      accreg_type = program->template device_type_name<int16_t>();
      break;
    case LIBDNN_ACCUMULATE_PREC_32:
      accreg_type = program->template device_type_name<int32_t>();
      break;
    case LIBDNN_ACCUMULATE_PREC_64:
      accreg_type = program->template device_type_name<int64_t>();
      break;
    default:
      break;
  }

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
  if (alpha_term) {
    args.push_back(program->template create_kernel_arg<MItype>("alpha",
                                                             KERNEL_ARG_CONST));
  }
  args.push_back(program->template create_kernel_arg<MItype>("A",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  args.push_back(program->template create_kernel_arg<MItype>("x",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  if (beta_term) {
    args.push_back(program->template create_kernel_arg<MItype>("beta",
                                                             KERNEL_ARG_CONST));
  }
  args.push_back(program->template create_kernel_arg<MOtype>("y",
                                  KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  ss << program->function("libdnn_gemv", args);

  // Thread identifiers
  // Local row ID (max: RTSM=TSM/WPTM)
  ss << "const int_tp tidn = " << program->local_id(0) << ";" << std::endl;
  // Work-group offset
  ss << "const int_tp offM = WPTM * " << program->global_id(0) << ";"
     << std::endl;

  // Local memory
  // xsub
  ss << program->local_mem("MItype", "xsub[" + std::to_string(tsn) + "]") << ";"
     << std::endl;

  ss << accreg_type << vwm << " yreg[WPTM / VWM];" << std::endl;

  // Initialize accumulator to 0
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm = 0; wm < WPTM; ++wm) {" << std::endl;
  if (unroll) {
    for (int m = 0; m < vwm; ++m) {
      ss << "VEC_" << vwm << "_" << m << "(yreg[wm]) = ("
         << accreg_type << ")0;" << std::endl;
    }
  } else {
    ss << "yreg[wm] = (" << accreg_type << ")0;" << std::endl;
  }
  ss << "}" << std::endl;

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
           << "x[tidn * WPTN + offN + " << n << "];" << std::endl;
        if (mode == 1) {
          ss << "} else {" << std::endl;
          ss << "xsub[tidn * WPTN + " << n << "] = "
             << "(MItype)0;" << std::endl;
          ss << "}" << std::endl;
        }
      }
      // Barrier after loading xsub
      ss << program->local_barrier() << std::endl;
      // Temporary registers for A and x
      ss << "MItype" << vwm << " Areg[WPTM][WPTN / VWN];" << std::endl;
      ss << "MItype" << vwn << " xreg[WPTN / VWN];" << std::endl;

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
      if (unroll || mode == 1) {
        for (int n = 0; n < vwn; ++n) {
          if (mode == 1) {
            ss << "if (wn + t * WPTN + offN + " << n << " < N) {" << std::endl;
          }
          if (trans_A) {
            ss << "VEC_" << vwn << "_" << n << "(Areg[wm][wn])"
               << " = A[(wn + t * WPTN + offN + " << n << ") * M + wm + offM];"
               << std::endl;
          } else {
            ss << "VEC_" << vwn << "_" << n << "(Areg[wm][wn])"
               << " = A[wn + t * WPTN + offN + " << n << " + (wm + offM) * N];"
               << std::endl;
          }
          if (mode == 1) {
            ss << "} else {" << std::endl;
            ss << "VEC_" << vwn << "_" << n << "(Areg[wm][wn]) = (MItype)0;"
               << std::endl;
            ss << "}" << std::endl;
          }
        }
      } else {
        if (trans_A) {
          ss << "Areg[wm][wn] = A[(wn + t * WPTN + offN) * M + (wm + offM)]"
             << std::endl;
        } else {
          ss << "Areg[wm][wn] = A[wn + t * WPTN + offN + (wm + offM) * N]"
             << std::endl;
        }
      }
      ss << "}" << std::endl;  // WPTN/VWN
      ss << "} else {" << std::endl;  // M-Guard
      ss << "for (int_tp wn = 0; wn < WPTN/VWN; ++wn) {" << std::endl;
      if (unroll) {
        for (int n = 0; n < vwn; ++n) {
          ss << "VEC_" << vwn << "_" << n << "(Areg[wm][wn]) = (MItype)0;"
             << std::endl;
        }
      } else {
        ss << "Areg[wm][wn] = (MItype)0;" << std::endl;
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
            if (prec != LIBDNN_ACCUMULATE_PREC_NATIVE) {
              ss << "(" << accreg_type << ")";
            }
            ss << "(VEC_" << vwn << "_" << n << "(Areg[wm * VWM + " << m
               << "][wn])" << " * VEC_" << vwn << "_" << n << "(xreg[wn]));"
               << std::endl;
          }
        }
      } else {
        for (int_tp m = 0; m < vwm; ++m) {
          ss << accreg_type << vwn << " tmp;" << std::endl;
          ss << "tmp = " << std::endl;
          stringstream src_term;
          src_term << " Areg[wm * VWM + " << m << "][wn] * xreg[wn]";
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
              ss << src_term.str() << ";" << std::endl;
              break;
          }
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

  // Store y
  ss << "for (int_tp wm = 0; wm < WPTM/VWM; ++wm) {" << std::endl;
  for (int_tp m = 0; m < vwm; ++m) {
    ss << "if (wm + offM + " << m << " < M) {" << std::endl;
    ss << "y[offM + wm + " << m << "] = ";
    if (alpha_term) {
      ss << "alpha * ";
    }
    ss << "VEC_" << vwm << "_" << m << "(yreg[wm])";
    if (beta_term) {
      ss << " + beta * y[offM + wm + " << m << "]";
    }
    ss << ";" << std::endl;
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
    bool alpha_term, bool beta_term, libdnnAccumulatePrecision_t prec,
    shared_ptr<Quantizer<MItype, MOtype> > quantizer) {
  stringstream ss;
  ss << "gemv_";
  ss << (trans_A == CblasNoTrans ? "TA_" : "NTA_");
  ss << "M" << M << "_";
  ss << "N" << N << "_";
  if (alpha_term) {
    ss << "alpha_";
  }
  if (beta_term) {
    ss << "beta_";
  }
  switch (prec) {
    case LIBDNN_ACCUMULATE_PREC_8:
      ss << "prec_8_";
      break;
    case LIBDNN_ACCUMULATE_PREC_16:
      ss << "prec_16_";
      break;
    case LIBDNN_ACCUMULATE_PREC_32:
      ss << "prec_32_";
      break;
    case LIBDNN_ACCUMULATE_PREC_64:
      ss << "prec_64_";
      break;
    default:
      break;
  }
  ss << "q_" << (quantizer->needs_quantization() ? "a" : "p");
  return ss.str();
}


template<typename MItype, typename MOtype>
void LibDNNBlas<MItype, MOtype>::gemv(
               const CBLAS_TRANSPOSE trans_A,
               const uint_tp M, const uint_tp N,
               const MItype alpha, vptr<const MItype> A, vptr<const MItype> x,
               const MItype beta, vptr<MOtype> y,
               libdnnAccumulatePrecision_t prec,
               shared_ptr<Quantizer<MItype, MOtype> > quantizer) {
  bool alpha_term = alpha != (MItype)1;
  bool beta_term = beta != (MItype)0;

  uint_tp MT = trans_A == CblasNoTrans ? M : N;
  uint_tp NT = trans_A == CblasNoTrans ? N : M;

  string identifier = gemv_string_identifier(trans_A, MT, NT,
                               alpha_term, beta_term, prec, quantizer);

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
                                 MT, NT, alpha_term, beta_term, prec,
                                 quantizer);
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

  if (alpha_term) {
    kernel->add_arg(&alpha);
  }
  kernel->add_arg(&A);
  kernel->add_arg(&x);
  if (beta_term) {
    kernel->add_arg(&beta);
  }
  kernel->add_arg(&y);
  kernel->Execute(group, local);
}

INSTANTIATE_CLASS_2T_GUARDED(LibDNNBlas, PROTO_TYPES, PROTO_TYPES);

}  // namespace caffe

#endif  // USE_LIBDNN
