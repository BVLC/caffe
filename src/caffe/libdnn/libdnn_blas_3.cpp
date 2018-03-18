#ifdef USE_LIBDNN

#include "caffe/common.hpp"
#include "caffe/libdnn/libdnn_blas.hpp"

namespace caffe {

template<typename MItype, typename MOtype>
void LibDNNBlas<MItype, MOtype>::initialize_gemm_tuner(
    shared_ptr<DeviceProgram> program, shared_ptr<LibDNNTuner> tuner) {
  // Work groups
  for (int id = 0; id < 2; ++id) {
    vector<int_tp> workgroup_sizes;
    for (int_tp i = 0; i < this->dev_ptr_->workgroup_size(id);
            i += 4) {
      workgroup_sizes.push_back(i);
    }
    tuner->add_set_param <int_tp>("workgroup_size_" + std::to_string(id),
                                      16, workgroup_sizes);
  }

  tuner->add_range_param<int_tp>("TSK", 8, 1, 32, 1);
  tuner->add_range_param<int_tp>("TSK_UNROLL", 1, 1, 16, 1);
  tuner->add_range_param<int_tp>("WPTM", 4, 4, 16, 4);
  tuner->add_set_param<int_tp>("VWM", 4, vector<int_tp>({1, 2, 4, 8, 16 }));
  tuner->add_range_param<int_tp>("WPTN", 4, 4, 16, 4);
  tuner->add_set_param<int_tp>("VWN", 4, vector<int_tp>({1, 2, 4, 8, 16 }));

  tuner->add_constraint<int64_t>(
    vector<string>({"TSK", "WPTM", "workgroup_size_1"}),
    vector<string>({"TSK"}), [](vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });

  tuner->add_constraint<int64_t>(
    vector<string>({"TSK", "WPTN", "workgroup_size_0"}),
    vector<string>({"TSK"}), [](vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });

  tuner->add_constraint<int64_t>(
    vector<string>({"TSK", "TSK_UNROLL"}),
    vector<string>({"TSK_UNROLL"}), [](vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });

  tuner->add_constraint<int64_t>(
    vector<string>({"WPTM", "VWM"}),
    vector<string>({"WPTM"}), [](vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });

  tuner->add_constraint<int64_t>(
    vector<string>({"WPTN", "VWN"}),
    vector<string>({"WPTN"}),
    [](vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });

  tuner->add_range_param<int_tp>("lmem_pad_A", 0, 0, 8, 1);
  tuner->add_range_param<int_tp>("lmem_pad_B", 0, 0, 8, 1);

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
string LibDNNBlas<MItype, MOtype>::generate_gemm_source(
    shared_ptr<DeviceProgram> program, shared_ptr<LibDNNTuner> tuner,
    bool trans_A, bool trans_B,
    const uint_tp M, const uint_tp N, const uint_tp K,
    bool alpha_term, bool beta_term) {

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
  ss << program->vector_accessors();

  string accreg_type = "MItype";

  int wptn = tuner->get_param<int>("WPTN");
  int wptm = tuner->get_param<int>("WPTM");
  int tsk = tuner->get_param<int>("TSK");
  int rtsn = tuner->get_param<int>("workgroup_size_0");
  int rtsm = tuner->get_param<int>("workgroup_size_1");
  int tsm = wptm * rtsm;
  int tsn = wptn * rtsn;
  int vwm = tuner->get_param<int>("VWM");
  int vwn = tuner->get_param<int>("VWN");
  int lpta = (tsm * tsk) / (rtsm * rtsn);
  int lptb = (tsn * tsk) / (rtsm * rtsn);


  // GEMM definitions
  ss << program->define("M", M);
  ss << program->define("N", N);
  ss << program->define("K", K);

  // Local memory padding
  ss << program->define("v_pad_A", tuner->get_param<int>("lmem_pad_A"));
  ss << program->define("v_pad_B", tuner->get_param<int>("lmem_pad_B"));

  // The tile-size in dimension M
  ss << program->define("TSM", tuner->get_param<int>("WPTM")
          * tuner->get_param<int>("workgroup_size_1"));
  // The tile-size in dimension N
  ss << program->define("TSN", tuner->get_param<int>("WPTN")
          * tuner->get_param<int>("workgroup_size_0"));
  // The tile-size in dimension K
  ss << program->define("TSK", tuner->get_param<int>("TSK"));
  // TSK unrolling
  ss << program->define("TSK_UNROLL",
                         tuner->get_param<int>("TSK_UNROLL"));
  // The work-per-thread in dimension M
  ss << program->define("WPTM", tuner->get_param<int>("WPTM"));
  ss << program->define("VWM", tuner->get_param<int>("VWM"));
  // The work-per-thread in dimension N
  ss << program->define("WPTN", tuner->get_param<int>("WPTN"));
  ss << program->define("VWN", tuner->get_param<int>("VWN"));
  // The reduced tile-size in dimension M
  ss << program->define("RTSM",
                         tuner->get_param<int>("workgroup_size_1"));
  // The reduced tile-size in dimension N
  ss << program->define("RTSN",
                         tuner->get_param<int>("workgroup_size_0"));
  // Loads-per-thread for A
  ss << program->define("LPTA", "((TSK*TSM)/(RTSM*RTSN))");
  // Loads-per-thread for B
  ss << program->define("LPTB", "((TSK*TSN)/(RTSM*RTSN))");

  // Num tiles needs to be next higher even integer
  // (due to some quirky bug in AMD OpenCL 2.0 on Windows)
  ss << program->define("v_num_tiles", "(((K - 1)/(TSK*2) + 1)*2)");

  KernelArgs args;
  if (alpha_term) {
    args.push_back(program->template create_kernel_arg<MItype>("alpha",
                                                             KERNEL_ARG_CONST));
  }
  args.push_back(program->template create_kernel_arg<MItype>("A",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  args.push_back(program->template create_kernel_arg<MItype>("B",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  if (beta_term) {
    args.push_back(program->template create_kernel_arg<MItype>("beta",
                                                             KERNEL_ARG_CONST));
  }
  args.push_back(program->template create_kernel_arg<MOtype>("C",
                                  KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  ss << program->function("libdnn_gemm", args);

  ss << program->global_ptr("const MItype", "Aptr") << " = A;"
     << std::endl;
  ss << program->global_ptr("const MItype", "Bptr")<< " = B;"
     << std::endl;
  ss << program->global_ptr("MOtype", "Cptr")<< " = C;"
     << std::endl;

  // Thread identifiers
  // Local row ID (max: RTSM=TSM/WPTM)
  ss << "const int_tp tidn = " << program->local_id(0) << ";"
     << std::endl;
  // Local col ID (max: RTSN=TSN/WPTN)
  ss << "const int_tp tidm = " << program->local_id(1) << ";"
     << std::endl;
  // Work-group offset
  ss << "const int_tp offN = TSN * " << program->group_id(0) << ";"
     << std::endl;
  // Work-group offset
  ss << "const int_tp offM = TSM * " << program->group_id(1) << ";"
     << std::endl;

  // Local tile memory
  // Asub
  ss << program->local_mem("MItype",
                      "Asub[" + std::to_string(tsm) + "]"
                      + "[" + std::to_string(tsk) + " + v_pad_A]") << ";"
                    << std::endl;
  // Bsub
  ss << program->local_mem("MItype",
                      "Bsub[" + std::to_string(tsk) + "]"
                      + "[" + std::to_string(tsn) + " + v_pad_B]") << ";"
                    << std::endl;

  // Initialize the accumulation registers
  ss << "{" << std::endl;  // Scoping for C registers
  ss << this->generate_accreg_init(tuner, false, beta_term, beta_term);

  ss << "{" << std::endl;  // Scoping for load & compute block
  // Loop over all tiles
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int_tp t = 0; t < v_num_tiles; ++t) {" << std::endl;

  // Load one tile of A into local memory
  ss << "{" << std::endl;  // Scoping for loading A
  ss << "#pragma unroll 4" << std::endl;
  ss << "for (int_tp la = 0; la < LPTA; ++la) {" << std::endl;
  ss << "int_tp tid = tidm * RTSN + tidn;" << std::endl;
  ss << "int_tp id = la * RTSN * RTSM + tid;" << std::endl;
  ss << "int_tp row = id / TSK;" << std::endl;
  ss << "int_tp col = id % TSK;" << std::endl;
  ss << "int_tp tiledIndex = TSK * t + col;" << std::endl;
  ss << "if ((offM + row) < M && tiledIndex < K) {" << std::endl;
  ss << "Asub[row][col] = ";
  if (trans_A) {
    ss << "Aptr[(offM + row) + tiledIndex * M];";
  } else {
    ss << "Aptr[(offM + row) * K + tiledIndex];";
  }
  ss << std::endl;
  ss << "} else {" << std::endl;  // M-K-Guard
  ss << "Asub[row][col] = (MItype)0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // LPTA
  ss << "}" << std::endl;  // Scoping for loading A

  // Load one tile of B into local memory
  ss << "{" << std::endl;  // Scoping for loading B
  ss << "#pragma unroll 4" << std::endl;
  ss << "for (int_tp lb = 0; lb < LPTB; ++lb) {" << std::endl;
  ss << "int_tp tid = tidm * RTSN + tidn;" << std::endl;
  ss << "int_tp id = lb * RTSN * RTSM + tid;" << std::endl;
  ss << "int_tp row = id / TSN;" << std::endl;
  ss << "int_tp col = id % TSN;" << std::endl;
  ss << "int_tp tiledIndex = TSK * t + row;" << std::endl;
  ss << "if ((offN + col) < N && tiledIndex < K) {" << std::endl;
  if (trans_B) {
    ss << "Bsub[row][col] = Bptr[(offN + col) * K + tiledIndex];"
       << std::endl;
  } else {
    ss << "Bsub[row][col] = Bptr[(offN + col) + tiledIndex * N];"
       << std::endl;
  }
  ss << "} else {" << std::endl;  // N-K-Guard
  ss << "Bsub[row][col] = (MItype)0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // LPTB
  ss << "}" << std::endl;  // Scoping for loading B

  // Synchronize to make sure the tile is loaded
  ss << program->local_barrier() << std::endl;

  ss << this->generate_gemm_core(tuner, false, alpha_term);

  ss << program->local_barrier() << std::endl;

  // Loop over all tiles
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for load & compute block

  // Store the final results in C
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
  ss << "int_tp globalRow = offM + tidm + wm * RTSM;"
     << std::endl;
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wn=0; wn<WPTN; ++wn) {" << std::endl;
  ss << "int_tp globalCol = offN + tidn + wn * RTSN;"
     << std::endl;
  ss << "if (globalRow < M && globalCol < N) {" << std::endl;
  ss << "Cptr[globalRow * N + globalCol] = ";
  ss << "(MOtype)";
  ss << "(((" << accreg_type << "*)(&(Creg[wm][wn/VWN])))[wn%VWN]);"
     << std::endl;
  ss << "}" << std::endl;   // M-N-Guard
  ss << "}" << std::endl;   // For (N)
  ss << "}" << std::endl;   // For (M)
  ss << "}" << std::endl;   // Scoping for C registers

  // Kernel
  ss << "}" << std::endl;
  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::gemm_string_identifier(
    const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
    const uint_tp M, const uint_tp N, const uint_tp K,
    bool alpha_term, bool beta_term) {
  stringstream ss;
  ss << "gemm_";
  ss << (trans_A == CblasNoTrans ? "TA_" : "NTA_");
  ss << (trans_B == CblasNoTrans ? "TB_" : "NTB_");
  ss << "M" << M << "_";
  ss << "N" << N << "_";
  ss << "K" << K << "_";
  if (alpha_term) {
    ss << "alpha_";
  }
  if (beta_term) {
    ss << "beta_";
  }
  return ss.str();
}

template<typename MItype, typename MOtype>
void LibDNNBlas<MItype, MOtype>::gemm(
               const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
               const uint_tp M, const uint_tp N, const uint_tp K,
               const MOtype alpha, vptr<const MItype> A, vptr<const MItype> B,
               const MOtype beta, vptr<MOtype> C,
               const QuantizerValues* const alpha_quant,
               const QuantizerValues* const a_quant,
               const QuantizerValues* const b_quant,
               const QuantizerValues* const beta_quant,
               const QuantizerValues* const c_quant) {

  typedef typename std::conditional<float_is_same<MItype>::value, MItype,
          typename std::conditional<sizeof(MItype) == 1, int16_t,
          typename std::conditional<sizeof(MItype) == 2, int32_t,
                                    int64_t>::type>::type>::type Difftype;
  typedef typename std::conditional<float_is_same<MItype>::value, MItype,
          typename std::conditional<sizeof(MItype) == 1, int32_t,
                                    int64_t>::type>::type Acctype;

  bool alpha_term = alpha != (MItype)1;
  bool beta_term = beta != (MItype)0;

  string identifier = gemm_string_identifier(trans_A, trans_B, M, N, K,
                                             alpha_term, beta_term);

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
      ss << generate_gemm_source(program, tuner,
                                 trans_A == CblasTrans, trans_B == CblasTrans,
                                 M, N, K, alpha_term, beta_term);
      program->set_source(ss.str());
      program->Compile(true, true);
      program_ready_[id] = true;
    }
    ulock.unlock();
    lock.lock();
  }
  lock.unlock();

  // Non-exclusive execute
  shared_ptr<DeviceKernel> kernel = program->GetKernel("libdnn_gemm");

  size_t wptn = tuner->get_param<int>("WPTN");
  size_t wptm = tuner->get_param<int>("WPTM");
  size_t wgs0 = tuner->get_param<int>("workgroup_size_0");
  size_t wgs1 = tuner->get_param<int>("workgroup_size_1");
  size_t div_N = wptn * wgs0;
  size_t div_M = wptm * wgs1;

  vector<size_t> group = {((N - 1) / div_N + 1),
                          ((M - 1) / div_M + 1),
                          1};
  vector<size_t> local = {wgs0, wgs1, 1};

  Acctype mult;
  int8_t shift;
  const Acctype c_max = c_quant ? c_quant->get_max<Acctype>() : Acctype(0);
  const Acctype c_min = c_quant ? c_quant->get_min<Acctype>() : Acctype(0);
  const Acctype result_offset = c_quant ? c_quant->get_zero<Acctype>()
      : Acctype(0);
  int8_t shift_bits = ((sizeof(Acctype))/sizeof(MItype)) - 1;
  MItype lhs_off = a_quant->get_zero<MItype>();
  MItype rhs_off = b_quant->get_zero<MItype>();
  QuantizerBase::template MultiplicativeQuantVals<Acctype>(
      a_quant, b_quant, c_quant, &mult, &shift, shift_bits);

  if (alpha_term) {
    kernel->add_arg(&alpha);
  }
  kernel->add_arg(&A);
  kernel->add_arg(&B);
  if (beta_term) {
    kernel->add_arg(&beta);
  }
  kernel->add_arg(&C);
  kernel->Execute(group, local);
}

INSTANTIATE_CLASS_2T_GUARDED(LibDNNBlas, PROTO_TYPES, PROTO_TYPES);

}  // namespace caffe

#endif  // USE_LIBDNN
