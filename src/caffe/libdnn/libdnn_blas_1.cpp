#ifdef USE_LIBDNN

#include "caffe/libdnn/libdnn_blas.hpp"

namespace caffe {

template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::scale_string_identifier(
    shared_ptr<Quantizer<MItype, MOtype> > quantizer) {
  stringstream ss;
  ss << "scale_";
  ss << "q_" << (quantizer->needs_quantization() ? "a" : "p");
  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::generate_scale_source(
            shared_ptr<DeviceProgram> program, shared_ptr<LibDNNTuner> tuner,
            shared_ptr<Quantizer<MItype, MOtype> > quantizer) {
  stringstream ss;
  KernelArgs args;
  args.push_back(program->create_kernel_arg<uint_tp>("n", KERNEL_ARG_CONST));
  args.push_back(program->create_kernel_arg<MItype>("alpha", KERNEL_ARG_CONST));
  args.push_back(program->create_kernel_arg<MItype>("x",
            KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM  | KERNEL_ARG_MEM_OFFSET));
  args.push_back(program->create_kernel_arg<MOtype>("y",
            KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_MEM_OFFSET));
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
           shared_ptr<Quantizer<MItype, MOtype> > quantizer) {
  string identifier = scale_string_identifier(quantizer);

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
      ss << generate_scale_source(program, tuner, quantizer);
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
          shared_ptr<Quantizer<MItype, MOtype> > quantizer) {
  this->scale(n, alpha, x, x, quantizer);
}

template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::axpby_string_identifier(
    shared_ptr<Quantizer<MItype, MOtype> > quantizer) {
  stringstream ss;
  ss << "axpby_";
  ss << "q_" << (quantizer->needs_quantization() ? "a" : "p");
  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::generate_axpby_source(
            shared_ptr<DeviceProgram> program, shared_ptr<LibDNNTuner> tuner,
            shared_ptr<Quantizer<MItype, MOtype> > quantizer) {
  stringstream ss;
  KernelArgs args;
  args.push_back(program->create_kernel_arg<uint_tp>("n", KERNEL_ARG_CONST));
  args.push_back(program->create_kernel_arg<MItype>("alpha", KERNEL_ARG_CONST));
  args.push_back(program->create_kernel_arg<MItype>("x",
            KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM  | KERNEL_ARG_MEM_OFFSET));
  args.push_back(program->create_kernel_arg<MItype>("beta", KERNEL_ARG_CONST));
  args.push_back(program->create_kernel_arg<MOtype>("y",
            KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_MEM_OFFSET));
  ss << program->function("libdnn_axpby", args);
  ss << program->kernel_loop("uint_tp", "index", "n");
  ss << "y[index] = alpha * x[index] + beta * y[index];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

template<typename MItype, typename MOtype>
void LibDNNBlas<MItype, MOtype>::axpby(const uint_tp n, const MItype alpha,
           vptr<const MItype> x, const MOtype beta, vptr<MOtype> y,
           shared_ptr<Quantizer<MItype, MOtype> > quantizer) {
  string identifier = axpby_string_identifier(quantizer);

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
      ss << generate_axpby_source(program, tuner,
                                  quantizer);
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
  kernel->add_arg(&n);
  kernel->add_arg(&alpha);
  kernel->add_arg(&x);
  kernel->add_arg(&beta);
  kernel->add_arg(&y);
  kernel->Execute(group, local);
}

template<typename MItype, typename MOtype>
string LibDNNBlas<MItype, MOtype>::dot_string_identifier(
    libdnnAccumulatePrecision_t prec,
    shared_ptr<Quantizer<MItype, MOtype> > quantizer) {
  stringstream ss;
  ss << "axpby_";
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
string LibDNNBlas<MItype, MOtype>::generate_dot_source(
                             shared_ptr<DeviceProgram> program,
                             shared_ptr<LibDNNTuner> tuner,
                             libdnnAccumulatePrecision_t prec,
                             shared_ptr<Quantizer<MItype, MOtype> > quantizer) {
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
         vptr<const MItype> y, MOtype* out, libdnnAccumulatePrecision_t prec,
         shared_ptr<Quantizer<MItype, MOtype> > quantizer) {
  string identifier = dot_string_identifier(prec, quantizer);

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
      ss << generate_dot_source(program, tuner, prec, quantizer);
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
string LibDNNBlas<MItype, MOtype>::asum_string_identifier(
    libdnnAccumulatePrecision_t prec,
    shared_ptr<Quantizer<MItype, MOtype> > quantizer) {
  stringstream ss;
  ss << "axpby_";
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
string LibDNNBlas<MItype, MOtype>::generate_asum_source(
                             shared_ptr<DeviceProgram> program,
                             shared_ptr<LibDNNTuner> tuner,
                             libdnnAccumulatePrecision_t prec,
                             shared_ptr<Quantizer<MItype, MOtype> > quantizer) {
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
  args.push_back(program->create_kernel_arg<MOtype>("out",
                                                   KERNEL_ARG_GLOBAL_MEM));
  ss << program->function("libdnn_asum", args);
  ss << "out = (MOtype) 0.0";
  ss << program->kernel_loop("uint_tp", "index", "n");
  ss << "out += x[index];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}


template<typename MItype, typename MOtype>
void LibDNNBlas<MItype, MOtype>::asum(const uint_tp n, vptr<const MItype> x,
          MOtype* y, libdnnAccumulatePrecision_t prec,
          shared_ptr<Quantizer<MItype, MOtype> > quantizer) {
  string identifier = asum_string_identifier(prec, quantizer);

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
      ss << generate_asum_source(program, tuner, prec, quantizer);
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
