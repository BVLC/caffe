#ifdef USE_LIBDNN

#include "caffe/libdnn/libdnn_blas.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
string LibDNNBlas<Dtype, MItype, MOtype>::create_gemm_source(
    shared_ptr<DeviceProgram> program, shared_ptr<LibDNNTuner> tuner,
    bool trans_A, bool trans_B,
    bool alpha_term, bool beta_term, LibDNNAccumulatePrecision prec) {
  stringstream ss;

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

  ss << program->setup();
  ss << program->atomics();
  ss << program->define_vector_type(program->device_type_name<Dtype>(), 0, 16);
  ss << program->define_vector_type(program->device_type_name<MItype>(), 0, 16);
  ss << program->define_vector_type(program->device_type_name<MOtype>(), 0, 16);
  ss << program->vector_accessors();

  // GEMM definitions
  ss << this->program_->define("MG", MG_FW_);
  ss << this->program_->define("M", M_FW_);
  ss << this->program_->define("N", N_FW_);
  ss << this->program_->define("KG", KG_FW_);
  ss << this->program_->define("K", K_FW_);

  // Local memory padding
  ss << this->program_->define("v_pad_A", tuner->get_param<int>("lmem_pad_A"));
  ss << this->program_->define("v_pad_B", tuner->get_param<int>("lmem_pad_B"));

  KernelArgs args;
  args.push_back(this->program_->template create_kernel_arg<Dtype>("alpha",
                                                             KERNEL_ARG_CONST));
  args.push_back(this->program_->template create_kernel_arg<MItype>("A",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  args.push_back(this->program_->template create_kernel_arg<MItype>("B",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  args.push_back(this->program_->template create_kernel_arg<Dtype>("beta",
                                                             KERNEL_ARG_CONST));
  args.push_back(this->program_->template create_kernel_arg<MOtype>("C",
                                  KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  ss << this->program_->function("libdnn_gemm", args);

  // Thread identifiers
  // Local row ID (max: RTSM=TSM/WPTM)
  ss << "const int_tp tidn = " << this->program_->local_id(0) << ";"
     << std::endl;
  // Local col ID (max: RTSN=TSN/WPTN)
  ss << "const int_tp tidm = " << this->program_->local_id(1) << ";"
     << std::endl;
  // Work-group offset
  ss << "const int_tp offN = TSN * " << this->program_->group_id(0) << ";"
     << std::endl;
  // Work-group offset
  ss << "const int_tp offM = TSM * " << this->program_->group_id(1) << ";"
     << std::endl;

  // Local tile memory
  // Asub for loading weights & shuffling the output
  ss << "volatile " << this->program_->local_mem("Dtype",
                      "Asub[" + std::to_string(tsm) + "]"
                      + "[" + std::to_string(tsk) + " + v_pad_A]") << ";"
                    << std::endl;
  // Bsub for loading the input image and shuffling the output image
  ss << "volatile " << this->program_->local_mem("Dtype",
                      "Bsub[" + std::to_string(tsk) + "]"
                      + "[" + std::to_string(tsn) + " + v_pad_B]") << ";"
                    << std::endl;

  // Initialize the accumulation registers
  ss << "{" << std::endl;  // Scoping for C registers
  ss << this->generate_accreg_init(fw_tuner_, false, beta_term, beta_term,
                                   );

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
  ss << "Asub[row][col] = (MItype)0.0;" << std::endl;
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

  ss << this->program_->local_barrier() << std::endl;
  ss << generate_gemm_core(tuner, false);
  ss << this->program_->local_barrier() << std::endl;

  // Loop over all tiles
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for load & compute block

  // Store the final results in C
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
  ss << "int_tp globalRow = offM + tidm + wm * RTSM;"
     << std::endl;
  if (bias_term_) {
    ss << "Dtype biasval = Dptr[globalRow];" << std::endl;
  }
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wn=0; wn<WPTN; ++wn) {" << std::endl;
  ss << "int_tp globalCol = offN + tidn + wn * RTSN;"
     << std::endl;
  ss << "if (globalRow < M && globalCol < N) {" << std::endl;
    ss << "Cptr[globalRow * N + globalCol] = ";
  if (alpha_term) {
    ss << "alpha * "
  }
  ss << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];" << std::endl;
  ss << "}" << std::endl;   // M-N-Guard
  ss << "}" << std::endl;   // For (N)
  ss << "}" << std::endl;   // For (M)
  ss << "}" << std::endl;   // Scoping for C registers

  // Kernel
  ss << "}" << std::endl;
  return ss.str();
}

template<typename Dtype, typename MItype, typename MOtype>
string LibDNNBlas<Dtype, MItype, MOtype>::gemm_string_identifier() {
  stringsteam ss;
  ss << "gemm_";
  return ss.str();
}


template<typename Dtype, typename MItype, typename MOtype>
void LibDNNBlas<Dtype, MItype, MOtype>::gemm(
                 const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
                 const uint_tp M, const uint_tp N, const uint_tp K,
                 const Dtype alpha, vptr<const MItype> A, vptr<const MItype> B,
                 const Dtype beta, vptr<MOtype> C,
                 shared_ptr<Quantizer<MItype, Dtype> > in_quantizer,
                 shared_ptr<Quantizer<Dtype, MOtype> > out_quantizer) {

  id = get_id(gemm_string_identifier());
  shared_ptr<LibDNNTuner> tuner = program_tuners_[id];
  shared_ptr<DeviceProgram> program = programs_[id];

  if (!program_ready_[id]) {
    stringstream ss;
    ss << generate_gemm_kernel();
    program->set_source(ss.str());
    program->Compile(true, true);
  }
  shared_ptr<DeviceKernel> kernel = this->program_->GetKernel("libdnn_gemm");
  vector<size_t> group = {((N - 1) / fw_div_N + 1),
                          ((M - 1) / fw_div_M + 1),
                          1};
  vector<size_t> local = {fw_wgs0, fw_wgs1, 1};

  kernel->add_arg(&alpha);
  kernel->add_arg(&A);
  kernel->add_arg(&B);
  kernel->add_arg(&beta);
  kernel->add_arg(&C);
  kernel->Execute(group, local);
}


INSTANTIATE_CLASS_3T(LibDNNBlas, (half_fp), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNNBlas, (float), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNNBlas, (double), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNNBlas, (int8_t), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNNBlas, (int16_t), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNNBlas, (int32_t), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNNBlas, (int64_t), VARIANT_TYPES, VARIANT_TYPES);

}  // namespace caffe

#endif  // USE_LIBDNN
