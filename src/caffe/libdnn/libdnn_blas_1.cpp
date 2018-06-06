#ifdef USE_LIBDNN

#include "caffe/libdnn/libdnn_blas.hpp"

namespace caffe {

template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::scale_string_identifier() {
  stringstream ss;
  ss << "scale";
  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::generate_scale_source(
            shared_ptr<DeviceProgram> program,
            shared_ptr<LibDNNTuner> tuner) {
  stringstream ss;
  ss << program->setup();
  ss << program->template define_vector_type<MItype>("MItype", 0, 16);
  ss << program->template define_vector_type<MOtype>("MOtype", 0, 16);
  ss << program->vector_accessors();
  KernelArgs args;
  args.push_back(program->create_kernel_arg<uint_tp>("n", KERNEL_ARG_CONST));
  args.push_back(program->create_kernel_arg<MItype>("alpha", KERNEL_ARG_CONST));
  args.push_back(program->create_kernel_arg<MItype>("x",
            KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM  | KERNEL_ARG_MEM_OFFSET));
  args.push_back(program->create_kernel_arg<MOtype>("y",
                                KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_MEM_OFFSET));
  ss << program->function("libdnn_scale", args);
  ss << program->kernel_loop("uint_tp", "index", "n");
  ss << "y[index] = alpha * x[index];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

template<typename MItype, typename MOtype>
void LibDNNBlas<MItype, MOtype>::scale(const uint_tp n,
           const MItype alpha, vptr<const MItype> x, vptr<MOtype> y,
           const QuantizerValues* const alpha_quant,
           const QuantizerValues* const x_quant,
           const QuantizerValues* const y_quant) {
  string identifier = scale_string_identifier();

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
      stringstream ss;
      ss << generate_scale_source(program, tuner);
      program->set_source(ss.str());
      program->Compile(true, true);
      program_ready_[id] = true;
    }
    ulock.unlock();
    lock.lock();
  }
  lock.unlock();

  shared_ptr<DeviceKernel> kernel = program->GetKernel("libdnn_scale");
  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->dev_ptr_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->add_arg(&n);
  kernel->add_arg(&alpha);
  kernel->add_arg(&x);
  kernel->add_arg(&y);
  kernel->Execute(group, local);
}

template<typename MItype, typename MOtype>
void LibDNNBlas<MItype, MOtype>::scal(const uint_tp n, const MItype alpha,
          vptr<MItype> x,
          const QuantizerValues* const alpha_quant,
          const QuantizerValues* const x_quant) {
  this->scale(n, alpha, x, x, alpha_quant, x_quant, x_quant);
}

template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::axpby_string_identifier() {
  stringstream ss;
  ss << "axpby";
  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::generate_axpby_source(
             shared_ptr<DeviceProgram> program, shared_ptr<LibDNNTuner> tuner) {

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

  KernelArgs args;
  if (is_integer_type<MItype>() || is_integer_type<MOtype>()) {
    args.push_back(program->template create_kernel_arg<int8_t>("shift_bits",
                                                             KERNEL_ARG_CONST));
  }
  args.push_back(program->create_kernel_arg<uint_tp>("n", KERNEL_ARG_CONST));
  args.push_back(program->create_kernel_arg<MItype>("alpha", KERNEL_ARG_CONST));
  if (is_integer_type<MItype>()) {
    args.push_back(program->template create_kernel_arg<MItype>("alpha_off",
                                                           KERNEL_ARG_CONST));
    args.push_back(program->template create_kernel_arg<int32_t>("alpha_mult",
                                                           KERNEL_ARG_CONST));
    args.push_back(program->template create_kernel_arg<int8_t>("alpha_shift",
                                                           KERNEL_ARG_CONST));
  }
  args.push_back(program->create_kernel_arg<MItype>("x",
            KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM  | KERNEL_ARG_MEM_OFFSET));
  if (is_integer_type<MItype>()) {
    args.push_back(program->template create_kernel_arg<MItype>("x_off",
                                                             KERNEL_ARG_CONST));
  }
  args.push_back(program->create_kernel_arg<MOtype>("beta", KERNEL_ARG_CONST));
  if (is_integer_type<MOtype>()) {
    args.push_back(program->template create_kernel_arg<MOtype>("beta_off",
                                                           KERNEL_ARG_CONST));
    args.push_back(program->template create_kernel_arg<int32_t>("beta_mult",
                                                           KERNEL_ARG_CONST));
    args.push_back(program->template create_kernel_arg<int8_t>("beta_shift",
                                                           KERNEL_ARG_CONST));
  }
  args.push_back(program->create_kernel_arg<MOtype>("y",
                                KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_MEM_OFFSET));
  if (is_integer_type<MOtype>()) {
    args.push_back(program->template create_kernel_arg<MOtype>("y_off",
                                                             KERNEL_ARG_CONST));
    args.push_back(program->template create_kernel_arg<Acctype>("y_min",
                                                             KERNEL_ARG_CONST));
    args.push_back(program->template create_kernel_arg<Acctype>("y_max",
                                                             KERNEL_ARG_CONST));
  }

  ss << program->function("libdnn_axpby", args);
  ss << program->kernel_loop("uint_tp", "index", "n");
  if (is_float_type<MItype>()) {
    ss << "y[index] = alpha * x[index] + beta * y[index];" << std::endl;
  } else {
    ss << "Difftype alpha_diff = (Difftype)alpha - (Difftype)alpha_off;"
       << std::endl;
    ss << "Difftype x_diff = (Difftype)(x[index]) - (Difftype)x_off;"
       << std::endl;
    ss << "Acctype x_tmp = ((Acctype)alpha_diff) * ((Acctype)x_diff);"
       << std::endl;
    ss << "x_tmp = (Acctype)((((Multtype)x_tmp) * ((Multtype)alpha_mult))"
       << "/ ((Multtype)1 << shift_bits));" << std::endl;
    ss << "if (alpha_shift >= 0) {" << std::endl;
    ss << "Acctype mask = ((Acctype)1 << alpha_shift) - (Acctype)1;" << std::endl;
    ss << "Acctype x_round = (x_tmp & mask) > "
       << "((mask >> 1) + ((x_tmp < (Acctype)0) ? (Acctype)1 : (Acctype)0)) ? "
       << "(Acctype)1 : (Acctype)0;" << std::endl;
    ss << "x_tmp = (x_tmp >> alpha_shift) + x_round;" << std::endl;
    ss << "} else {" << std::endl;
    ss << "x_tmp = x_tmp << -alpha_shift;" << std::endl;
    ss << "}" << std::endl;

    ss << "Difftype beta_diff = (Difftype)beta - (Difftype)beta_off;"
       << std::endl;
    ss << "Difftype y_diff = (Difftype)(y[index]) - (Difftype)y_off;"
       << std::endl;
    ss << "Acctype y_tmp = ((Acctype)beta_diff) * ((Acctype)y_diff);"
       << std::endl;
    ss << "y_tmp = (Acctype)((((Multtype)y_tmp) * ((Multtype)beta_mult))"
       << "/ ((Multtype)1 << shift_bits));" << std::endl;
    ss << "if (beta_shift >= 0) {" << std::endl;
    ss << "Acctype mask = ((Acctype)1 << beta_shift) - (Acctype)1;" << std::endl;
    ss << "Acctype y_round = (y_tmp & mask) > "
       << "((mask >> 1) + ((y_tmp < (Acctype)0) ? (Acctype)1 : (Acctype)0)) ? "
       << "(Acctype)1 : (Acctype)0;" << std::endl;
    ss << "y_tmp = (y_tmp >> beta_shift) + y_round;" << std::endl;
    ss << "} else {" << std::endl;
    ss << "y_tmp = y_tmp << -beta_shift;" << std::endl;
    ss << "}" << std::endl;

    ss << "y[index] = (MOtype)(min(max(x_tmp + y_tmp + (Acctype)y_off, y_min), "
       << "y_max));"  << std::endl;
  }
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

template<typename MItype, typename MOtype>
void LibDNNBlas<MItype, MOtype>::axpby(const uint_tp n, const MItype alpha,
           vptr<const MItype> x, const MOtype beta, vptr<MOtype> y,
           const QuantizerValues* const alpha_quant,
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
  string identifier = axpby_string_identifier();

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
      stringstream ss;
      ss << generate_axpby_source(program, tuner);
      program->set_source(ss.str());
      program->Compile(true, true);
      program_ready_[id] = true;
    }
    ulock.unlock();
    lock.lock();
  }
  lock.unlock();

  shared_ptr<DeviceKernel> kernel = program->GetKernel("libdnn_axpby");
  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->dev_ptr_->get_threads(&work_size, &group, &local, kernel.get(), true);

  // Quantization parameters
  int8_t shift_bits =
      (this->dev_ptr_->template preferred_vector_width<int64_t>() > 0 ? 32 : 16)
      / sizeof(MItype) - 1;

  MItype x_off;
  MOtype y_off;
  Acctype y_min;
  Acctype y_max;
  MItype alpha_off;
  int32_t alpha_mult;
  int8_t alpha_shift;
  MOtype beta_off;
  int32_t beta_mult;
  int8_t beta_shift;

  if (is_integer_type<MItype>() || is_integer_type<MOtype>()) {
    x_off = x_quant->get_zero<MItype>();
    y_off = y_quant->get_zero<MOtype>();
    y_min = y_quant->get_min<Acctype>();
    y_max = y_quant->get_max<Acctype>();

    QuantizerValues qv_alpha;
    if (!alpha_quant) {
      qv_alpha.max = 1.0;
      qv_alpha.min = 0.0;
      qv_alpha.one = 1.0;
      qv_alpha.zero = 0.0;
      qv_alpha.scale = 1.0;
    }

    QuantizerValues qv_beta;
    if (!beta_quant) {
      qv_beta.max = 1.0;
      qv_beta.min = 0.0;
      qv_beta.one = 1.0;
      qv_beta.zero = 0.0;
      qv_beta.scale = 1.0;
    }

    alpha_off = alpha_quant ? alpha_quant->get_zero<MItype>()
                            : qv_alpha.get_zero<MItype>();
    beta_off = beta_quant ? beta_quant->get_zero<MOtype>()
                          : qv_beta.get_zero<MOtype>();

    QuantizerBase::template MultiplicativeQuantVals<int32_t>(
        x_quant, alpha_quant ? alpha_quant : &qv_alpha,
        y_quant, &alpha_mult, &alpha_shift, shift_bits);
    QuantizerBase::template MultiplicativeQuantVals<int32_t>(
        y_quant, beta_quant ? beta_quant : &qv_beta,
        y_quant, &beta_mult, &beta_shift, shift_bits);
  }

  if (is_integer_type<MItype>() || is_integer_type<MOtype>()) {
    kernel->add_arg(&shift_bits);
  }
  kernel->add_arg(&n);
  kernel->add_arg(&alpha);
  if (is_integer_type<MItype>()) {
    kernel->add_arg(&alpha_off);
    kernel->add_arg(&alpha_mult);
    kernel->add_arg(&alpha_shift);
  }
  kernel->add_arg(&x);
  if (is_integer_type<MItype>()) {
    kernel->add_arg(&x_off);
  }
  kernel->add_arg(&beta);
  if (is_integer_type<MOtype>()) {
    kernel->add_arg(&beta_off);
    kernel->add_arg(&beta_mult);
    kernel->add_arg(&beta_shift);
  }
  kernel->add_arg(&y);
  if (is_integer_type<MOtype>()) {
    kernel->add_arg(&y_off);
    kernel->add_arg(&y_min);
    kernel->add_arg(&y_max);
  }
  kernel->Execute(group, local);
}

template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::dot_string_identifier() {
  stringstream ss;
  ss << "dot";
  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::generate_dot_source(
                             shared_ptr<DeviceProgram> program,
                             shared_ptr<LibDNNTuner> tuner) {
  stringstream ss;
  ss << program->setup();
  ss << program->template define_vector_type<MItype>("MItype", 0, 16);
  ss << program->template define_vector_type<MOtype>("MOtype", 0, 16);
  ss << program->vector_accessors();
  // TODO: Better implementation
  KernelArgs args;
  args.push_back(program->create_kernel_arg<uint_tp>("n",
                    KERNEL_ARG_CONST));
  args.push_back(program->create_kernel_arg<MItype>("x",
                    KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM
                    | KERNEL_ARG_MEM_OFFSET));
  args.push_back(program->create_kernel_arg<MItype>("y",
                    KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM
                    | KERNEL_ARG_MEM_OFFSET));
  args.push_back(program->create_kernel_arg<MOtype>("out",
                    KERNEL_ARG_NONE));
  ss << program->function("libdnn_dot", args);
  ss << program->kernel_loop("uint_tp", "index", "n");
  ss << "out += x[index] * y[index];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}


template<typename MItype, typename MOtype>
void LibDNNBlas<MItype, MOtype>::dot(const uint_tp n, vptr<const MItype> x,
         vptr<const MItype> y, MOtype* out,
         const QuantizerValues* const x_quant,
         const QuantizerValues* const y_quant,
         const QuantizerValues* const out_quant) {
  string identifier = dot_string_identifier();

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
      stringstream ss;
      ss << generate_dot_source(program, tuner);
      program->set_source(ss.str());
      program->Compile(true, true);
      program_ready_[id] = true;
    }
    ulock.unlock();
    lock.lock();
  }
  lock.unlock();

  int_tp buffer_id = -1;
  vector<int_tp> buffer_shape(1,1);
  shared_ptr<Blob<MOtype> > buff = this->dev_ptr_->template
                                       Buffer<MOtype>(buffer_shape, &buffer_id);
  vptr<MOtype> gpu_out = buff->mutable_gpu_data();

  shared_ptr<DeviceKernel> kernel = program->GetKernel("libdnn_dot");
  vector<size_t> group(1, 1);
  vector<size_t> local(1, 1);
  kernel->add_arg(&n);
  kernel->add_arg(&x);
  kernel->add_arg(&y);
  kernel->add_arg(&gpu_out);
  kernel->Execute(group, local);
  this->dev_ptr_->template copy<MOtype>(1, gpu_out, out);
  this->dev_ptr_->unlock_buffer(&buffer_id);
}


template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::asum_string_identifier() {
  stringstream ss;
  ss << "asum";
  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::generate_asum_source(
                             shared_ptr<DeviceProgram> program,
                             shared_ptr<LibDNNTuner> tuner) {
  stringstream ss;
  ss << program->setup();
  ss << program->template define_vector_type<MItype>("MItype", 0, 16);
  ss << program->template define_vector_type<MOtype>("MOtype", 0, 16);
  ss << program->vector_accessors();
  ss << program->template helper_functions<MItype>();
  ss << program->template helper_functions<MOtype>();
  // TODO: Better implementation
  KernelArgs args;
  args.push_back(program->create_kernel_arg<uint_tp>("n",
                    KERNEL_ARG_CONST));
  args.push_back(program->create_kernel_arg<MItype>("x",
                    KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM
                    | KERNEL_ARG_MEM_OFFSET));
  args.push_back(program->create_kernel_arg<MOtype>("out",
                                                   KERNEL_ARG_GLOBAL_MEM));
  ss << program->function("libdnn_asum", args);
  ss << "out[0] = (MOtype)0;" << std::endl;
  ss << program->kernel_loop("uint_tp", "index", "n");
  ss << "out[0] += abs(x[index]);" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}


template<typename MItype, typename MOtype>
void LibDNNBlas<MItype, MOtype>::asum(const uint_tp n, vptr<const MItype> x,
                                      MOtype* y,
                                      const QuantizerValues* const x_quant,
                                      const QuantizerValues* const y_quant) {
  string identifier = asum_string_identifier();

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
      stringstream ss;
      ss << generate_asum_source(program, tuner);
      program->set_source(ss.str());
      program->Compile(true, true);
      program_ready_[id] = true;
    }
    ulock.unlock();
    lock.lock();
  }
  lock.unlock();

  int_tp buffer_id = -1;
  vector<int_tp> buffer_shape(1, 1);
  shared_ptr<Blob<MOtype> > buff = this->dev_ptr_->template
                                       Buffer<MOtype>(buffer_shape, &buffer_id);
  vptr<MOtype> gpu_out = buff->mutable_gpu_data();

  shared_ptr<DeviceKernel> kernel = program->GetKernel("libdnn_asum");
  vector<size_t> group(1, 1);
  vector<size_t> local(1, 1);
  kernel->add_arg(&n);
  kernel->add_arg(&x);
  kernel->add_arg(&gpu_out);
  kernel->Execute(group, local);
  this->dev_ptr_->template copy<MOtype>(1, gpu_out, y);
  this->dev_ptr_->unlock_buffer(&buffer_id);
}


INSTANTIATE_CLASS_2T_GUARDED(LibDNNBlas, PROTO_TYPES, PROTO_TYPES);

}  // namespace caffe

#endif  // USE_LIBDNN
