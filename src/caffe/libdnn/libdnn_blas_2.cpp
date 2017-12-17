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
  tuner->add_set_param<int_tp>("VWM", 2, vector<int_tp>({1, 2, 4, 8, 16 }));
  tuner->add_set_param<int_tp>("VWN", 2, vector<int_tp>({1, 2, 4, 8, 16 }));

  tuner->add_constraint<int64_t>(
    vector<string>({"WPTM", "VWM"}),
    vector<string>({"WPTM"}), [](vector<int64_t> args) -> bool {
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
    shared_ptr<Quantizer<MItype, MItype> > in_quantizer,
    shared_ptr<Quantizer<MItype, MOtype> > out_quantize) {
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
  int vwm = tuner->get_param<int>("VWM");
  int vwn = tuner->get_param<int>("VWN");
  int rtsn = tuner->get_param<int>("workgroup_size_0");

  // Split up calculation if vector x is not divisible by rtsn
  bool needs_split = (N % rtsn != 0);
  // Check if the big block computation is needed at all
  bool needs_block = (N >= rtsn);

  // GEMV definitions
  ss << program->define("M", M);
  ss << program->define("N", N);

  // The reduced tile-size in dimension N
  ss << program->define("RTSN", tuner->get_param<int>("workgroup_size_0"));

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

  // xsub
  ss << "volatile " << program->local_mem("MItype",
                      "xsub[" + std::to_string(rtsn) + "]") << ";"
                    << std::endl;

  ss << accreg_type << vwm << " yreg[" << (wptm / vwm) << "];";
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm = 0; wm < WPTM / VWM; ++wm) {" << std::endl;

  // Initialize accumulator to 0
  for (int i = 0; i < vwm; ++i) {
    ss << "VEC_" << vwm << "_" << i << "(Areg)"
       << " = ((" << accreg_type << ")0);";
  }

  if (needs_block) {
    ss << "for (int_tp wn = 0; wn < RTSN - (N % RTSN); ++wn) {" << std::endl;
    ss << program->local_barrier() << std::endl;

    ss << program->local_barrier() << std::endl;
    ss << "}" << std::endl;
  }

  if (needs_split && needs_block) {
  }

  if (needs_split) {

  } else {

  }

  // Kernel
  ss << "}" << std::endl;
  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::gemv_string_identifier(
    const CBLAS_TRANSPOSE trans_A, const uint_tp M, const uint_tp N,
    bool alpha_term, bool beta_term, libdnnAccumulatePrecision_t prec,
    shared_ptr<Quantizer<MItype, MItype> > in_quantizer,
    shared_ptr<Quantizer<MItype, MOtype> > out_quantizer) {
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
  ss << "iq_" << in_quantizer->get_mode_string() << "_";
  ss << "oq_" << out_quantizer->get_mode_string();
  return ss.str();
}


template<typename MItype, typename MOtype>
void LibDNNBlas<MItype, MOtype>::gemv(
               const CBLAS_TRANSPOSE trans_A,
               const uint_tp M, const uint_tp N,
               const MItype alpha, vptr<const MItype> A, vptr<const MItype> x,
               const MItype beta, vptr<MOtype> y,
               libdnnAccumulatePrecision_t prec,
               shared_ptr<Quantizer<MItype, MItype> > in_quantizer,
               shared_ptr<Quantizer<MItype, MOtype> > out_quantizer) {
  bool alpha_term = alpha != (MItype)1;
  bool beta_term = beta != (MItype)0;

  string identifier = gemv_string_identifier(trans_A, M, N,
                                             alpha_term, beta_term, prec,
                                             in_quantizer, out_quantizer);

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
      initialize_gemm_tuner(program, tuner);
      stringstream ss;
      ss << generate_gemv_source(program, tuner,
                                 trans_A == CblasTrans,
                                 M, N, alpha_term, beta_term, prec,
                                 in_quantizer, out_quantizer);
      program->set_source(ss.str());
      program->Compile(true, true);
      program_ready_[id] = true;
    }
    ulock.unlock();
  }
  lock.unlock();

  shared_ptr<DeviceKernel> kernel = program->GetKernel("libdnn_gemv");

  size_t wptm = tuner->get_param<int>("WPTM");
  size_t wgs0 = tuner->get_param<int>("workgroup_size_0");
  size_t div_N = wgs0;
  size_t div_M = wptm;

  vector<size_t> group = {((N - 1) / div_N + 1),
                          ((M - 1) / div_M + 1),
                          1};
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
