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
  args.push_back(program->create_kernel_arg<MItype>("beta", KERNEL_ARG_CONST));
  args.push_back(program->create_kernel_arg<MOtype>("y",
                                KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_MEM_OFFSET));
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
           const QuantizerValues* const alpha_quant,
           const QuantizerValues* const x_quant,
           const QuantizerValues* const beta_quant,
           const QuantizerValues* const y_quant) {
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
  kernel->add_arg(&n);
  kernel->add_arg(&alpha);
  kernel->add_arg(&x);
  kernel->add_arg(&beta);
  kernel->add_arg(&y);
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
