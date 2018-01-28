#include <cmath>

#include "caffe/quantizer.hpp"
#include "caffe/backend/device.hpp"
#include "caffe/backend/device_program.hpp"
#include "caffe/backend/device_kernel.hpp"

namespace caffe {

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::GenerateKernels() {
  this->program_ = this->device_->CreateProgram();
  stringstream ss;
  ss << this->program_->setup();
  ss << this->program_->template define_type<MItype>("MItype");
  ss << this->program_->template define_type<MOtype>("MOtype");


  // Quantizer forward
  {
    KernelArgs args;
    args.push_back(this->program_->template create_kernel_arg<size_t>("n",
                                                             KERNEL_ARG_CONST));
    args.push_back(this->program_->template create_kernel_arg<MItype>("in",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST |
               KERNEL_ARG_MEM_OFFSET));
    args.push_back(this->program_->template create_kernel_arg<MOtype>("out",
          KERNEL_ARG_MEM_OFFSET | KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM));
    if (fw_scale_before_cast()) {
      args.push_back(this->program_->template create_kernel_arg<MItype>("scal",
                                                             KERNEL_ARG_CONST));
    } else {
      args.push_back(this->program_->template create_kernel_arg<MOtype>("scal",
                                                             KERNEL_ARG_CONST));
    }
    ss << this->program_->function("quantizer_forward", args);
    ss << this->program_->kernel_loop("uint_tp", "i", "n");
    ss << "out[i] = " << fw_scale_term(0, "scal", "in[i]") << ";" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  // Quantizer backward
  {
    KernelArgs args;
    args.push_back(this->program_->template create_kernel_arg<size_t>("n",
                                                             KERNEL_ARG_CONST));
    args.push_back(this->program_->template create_kernel_arg<MOtype>("in",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST |
               KERNEL_ARG_MEM_OFFSET));
    args.push_back(this->program_->template create_kernel_arg<MItype>("out",
          KERNEL_ARG_MEM_OFFSET | KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM));
    if (bw_scale_before_cast()) {
      args.push_back(this->program_->template create_kernel_arg<MOtype>("scal",
                                                             KERNEL_ARG_CONST));
    } else {
      args.push_back(this->program_->template create_kernel_arg<MItype>("scal",
                                                             KERNEL_ARG_CONST));
    }
    ss << this->program_->function("quantizer_backward", args);
    ss << this->program_->kernel_loop("uint_tp", "i", "n");
    ss << "out[i] = " << bw_scale_term(0, "scal", "in[i]") << ";" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  // Observe in
  {
    KernelArgs args;
    args.push_back(this->program_->template create_kernel_arg<size_t>("n",
                                                             KERNEL_ARG_CONST));
    args.push_back(this->program_->template create_kernel_arg<MItype>("data",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST |
               KERNEL_ARG_MEM_OFFSET));
    if (is_signed_integer_type<MItype>()) {
      args.push_back(this->program_->template create_kernel_arg<float>("scal",
                                                             KERNEL_ARG_CONST));
    }
    args.push_back(this->program_->template create_kernel_arg<float>(
        "inter_min", KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->program_->template create_kernel_arg<float>(
        "inter_max", KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM));
    ss << this->program_->function("quantizer_observe_in", args);
    ss << this->program_->local_mem("float", "local_min[64]") << ";"
       << std::endl;
    ss << this->program_->local_mem("float", "local_max[64]") << ";"
       << std::endl;
    ss << "local_min[" << this->program_->local_id(0) << "] = FLT_MAX;"
       << std::endl;
    ss << "local_max[" << this->program_->local_id(0) << "] = -FLT_MAX;"
       << std::endl;
    ss << this->program_->kernel_loop("uint_tp", "i", "n");
    if (is_signed_integer_type<MItype>()) {
      ss << "float value = ((float)(data[i]))/scal;" << std::endl;
    } else {
      ss << "float value = data[i];" << std::endl;
    }
    ss << "if (local_min[" << this->program_->local_id(0) << "] > "
       << "value) {" << std::endl;
    ss << "local_min[" << this->program_->local_id(0) << "] = value;"
       << std::endl;
    ss << "}" << std::endl;
    ss << "if (local_max[" << this->program_->local_id(0) << "] < "
       << "value) {" << std::endl;
    ss << "local_max[" << this->program_->local_id(0) << "] = value;"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;  // Kernel loop
    // Reduction
    ss << this->program_->local_barrier() << std::endl;
    ss << "uint_tp i = 32;" << std::endl;
    ss << "while (i != 0) {" << std::endl;
    ss << "if(" << this->program_->local_id(0) << " < i) {" << std::endl;
    ss << "if (local_min[" << this->program_->local_id(0) << "] > "
       << "local_min[" << this->program_->local_id(0) << " + i]) {"
       << std::endl;
    ss << "local_min[" << this->program_->local_id(0) << "] = "
       << "local_min[" << this->program_->local_id(0) << " + i];" << std::endl;
    ss << "}" << std::endl;
    ss << "if (local_max[" << this->program_->local_id(0) << "] < "
       << "local_max[" << this->program_->local_id(0) << " + i]) {"
       << std::endl;
    ss << "local_max[" << this->program_->local_id(0) << "] = "
       << "local_max[" << this->program_->local_id(0) << " + i];" << std::endl;
    ss << "}" << std::endl;    ss << "}" << std::endl;
    ss << this->program_->local_barrier() << std::endl;
    ss << "i /= 2;" << std::endl;
    ss << "}" << std::endl;  // while (i != 0)
    // Write partially reduced output
    ss << "if (" << this->program_->local_id(0) << " == 0 ) {" << std::endl;
    ss << "inter_min[" << this->program_->group_id(0) << "] = local_min[0];"
       << std::endl;
    ss << "inter_max[" << this->program_->group_id(0) << "] = local_max[0];"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  // Observe out
  {
    KernelArgs args;
    args.push_back(this->program_->template create_kernel_arg<size_t>("n",
                                                             KERNEL_ARG_CONST));
    args.push_back(this->program_->template create_kernel_arg<MOtype>("data",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST |
               KERNEL_ARG_MEM_OFFSET));
    if (is_signed_integer_type<MOtype>()) {
      args.push_back(this->program_->template create_kernel_arg<float>("scal",
                                                             KERNEL_ARG_CONST));
    }
    args.push_back(this->program_->template create_kernel_arg<float>(
        "inter_min", KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->program_->template create_kernel_arg<float>(
        "inter_max", KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM));
    ss << this->program_->function("quantizer_observe_out", args);
    ss << this->program_->local_mem("float", "local_min[64]") << ";"
       << std::endl;
    ss << this->program_->local_mem("float", "local_max[64]") << ";"
       << std::endl;
    ss << "local_min[" << this->program_->local_id(0) << "] = FLT_MAX;"
       << std::endl;
    ss << "local_max[" << this->program_->local_id(0) << "] = -FLT_MAX;"
       << std::endl;
    ss << this->program_->kernel_loop("uint_tp", "i", "n");
    if (is_signed_integer_type<MOtype>()) {
      ss << "float value = ((float)(data[i]))/scal;" << std::endl;
    } else {
      ss << "float value = data[i];" << std::endl;
    }
    ss << "if (local_min[" << this->program_->local_id(0) << "] > "
       << "value) {" << std::endl;
    ss << "local_min[" << this->program_->local_id(0) << "] = value;"
       << std::endl;
    ss << "}" << std::endl;
    ss << "if (local_max[" << this->program_->local_id(0) << "] < "
       << "value) {" << std::endl;
    ss << "local_max[" << this->program_->local_id(0) << "] = value;"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;  // Kernel loop
    // Reduction
    ss << this->program_->local_barrier() << std::endl;
    ss << "uint_tp i = 32;" << std::endl;
    ss << "while (i != 0) {" << std::endl;
    ss << "if(" << this->program_->local_id(0) << " < i) {" << std::endl;
    ss << "if (local_min[" << this->program_->local_id(0) << "] > "
       << "local_min[" << this->program_->local_id(0) << " + i]) {"
       << std::endl;
    ss << "local_min[" << this->program_->local_id(0) << "] = "
       << "local_min[" << this->program_->local_id(0) << " + i];" << std::endl;
    ss << "}" << std::endl;
    ss << "if (local_max[" << this->program_->local_id(0) << "] < "
       << "local_max[" << this->program_->local_id(0) << " + i]) {"
       << std::endl;
    ss << "local_max[" << this->program_->local_id(0) << "] = "
       << "local_max[" << this->program_->local_id(0) << " + i];" << std::endl;
    ss << "}" << std::endl;    ss << "}" << std::endl;
    ss << this->program_->local_barrier() << std::endl;
    ss << "i /= 2;" << std::endl;
    ss << "}" << std::endl;  // while (i != 0)
    // Write partially reduced output
    ss << "if (" << this->program_->local_id(0) << " == 0 ) {" << std::endl;
    ss << "inter_min[" << this->program_->group_id(0) << "] = local_min[0];"
       << std::endl;
    ss << "inter_max[" << this->program_->group_id(0) << "] = local_max[0];"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  this->program_->set_source(ss.str());
  this->program_->Compile(true, true);
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Forward_gpu(size_t n, vptr<const MItype> input,
                         vptr<MOtype> output) {
  this->quant_mutex_.lock();
  if (!program_ready_) {
    this->GenerateKernels();
  }
  this->quant_mutex_.unlock();

  MItype scal_before = this->needs_quantization() ?
      fw_scale_before_cast_val() : MItype(1);
  MOtype scal_after = this->needs_quantization() ?
      fw_scale_after_cast_val() : MOtype(1);

  shared_ptr<DeviceKernel> kernel =
                          this->program_->GetKernel("quantizer_forward");
  kernel->add_arg(&n);
  kernel->add_arg(&input);
  kernel->add_arg(&output);
  if (fw_scale_before_cast()) {
    kernel->add_arg(&scal_before);
  } else {
    kernel->add_arg(&scal_after);
  }

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Forward_gpu(size_t n, vptr<const void> input,
                         vptr<void> output) {
  this->Forward_gpu(n, vptr<const MItype>(input), vptr<MOtype>(output));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Forward_gpu(Blob<MItype>* input,
                                            Blob<MOtype>* output,
                                            bool fw_data,
                                            bool fw_diff) {
  CHECK_EQ(input->count(), output->count());
  if (fw_data) {
    this->Forward_gpu(input->count(),
                      input->gpu_data(),
                      output->mutable_gpu_data());
  }
  if (fw_diff) {
    this->Forward_gpu(input->count(),
                      input->gpu_diff(),
                      output->mutable_gpu_diff());
  }
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Backward_gpu(size_t n, vptr<const MOtype> input,
                         vptr<MItype> output) {
  this->quant_mutex_.lock();
  if (!program_ready_) {
    this->GenerateKernels();
  }
  this->quant_mutex_.unlock();

  MOtype scal_before = this->needs_quantization() ?
      bw_scale_before_cast_val() : MOtype(1);
  MItype scal_after = this->needs_quantization() ?
      bw_scale_after_cast_val() : MItype(1);

  shared_ptr<DeviceKernel> kernel =
                                this->program_->GetKernel("quantizer_backward");
  kernel->add_arg(&n);
  kernel->add_arg(&input);
  kernel->add_arg(&output);
  if (bw_scale_before_cast()) {
    kernel->add_arg(&scal_before);
  } else {
    kernel->add_arg(&scal_after);
  }

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Backward_gpu(size_t n, vptr<const void> input,
                         vptr<void> output) {
  this->Backward_gpu(n, vptr<const MOtype>(input), vptr<MItype>(output));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Backward_gpu(Blob<MOtype>* input,
                                             Blob<MItype>* output,
                                             bool bw_data,
                                             bool bw_diff) {
  CHECK_EQ(input->count(), output->count());
  if (bw_data) {
    this->Backward_gpu(input->count(),
                       input->gpu_data(),
                       output->mutable_gpu_data());
  }
  if (bw_diff) {
    this->Backward_gpu(input->count(),
                       input->gpu_diff(),
                       output->mutable_gpu_diff());
  }
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Observe_in_gpu(size_t n,
                                               vptr<const void> data) {
  Observe_in_gpu(n, vptr<const MItype>(data));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Observe_in_gpu(size_t n,
                                               vptr<const MItype> data) {
  if (mode_ == PASSIVE) {
    return;
  }
  this->quant_mutex_.lock();
  if (!program_ready_) {
    this->GenerateKernels();
  }
  this->quant_mutex_.unlock();

  float scal = std::max(std::abs(max_in_), std::abs(min_in_))
                             / std::max(std::abs(flt_max_), std::abs(flt_min_));

  vector<size_t> local(1, 64);
  vector<size_t> group(1, (n - 1) / 64 + 1);

  int_tp min_buffer_lock_id = -1;
  int_tp max_buffer_lock_id = -1;

  shared_ptr<Blob<float> > min_blob =
      this->device_->template Buffer<float>(vector<int_tp>(1, group[1]),
                            &min_buffer_lock_id);
  shared_ptr<Blob<float> > max_blob =
      this->device_->template Buffer<float>(vector<int_tp>(1, group[1]),
                            &max_buffer_lock_id);

  vptr<float> min_gpu_data = min_blob->mutable_gpu_data();
  this->device_->template set<float>(n, type_max_val<float>(), min_gpu_data);
  vptr<float> max_gpu_data = max_blob->mutable_gpu_data();
  this->device_->template set<float>(n, type_min_val<float>(), max_gpu_data);

  shared_ptr<DeviceKernel> kernel =
                          this->program_->GetKernel("quantizer_observe_in");
  kernel->add_arg(&n);
  kernel->add_arg(&data);
  if (is_signed_integer_type<MItype>()) {
    kernel->add_arg(&scal);
  }
  kernel->add_arg(&min_gpu_data);
  kernel->add_arg(&max_gpu_data);
  kernel->Execute(group, local);

  const float* min_cpu_data = min_blob->cpu_data();
  const float* max_cpu_data = max_blob->cpu_data();

  this->quant_mutex_.lock();
  for (size_t i = 0; i < group[0]; ++i) {
    observed_max_ = std::max(static_cast<double>(min_cpu_data[i]),
                             observed_min_);
    observed_min_ = std::min(static_cast<double>(max_cpu_data[i]),
                             observed_max_);
  }
  this->quant_mutex_.unlock();

  this->device_->unlock_buffer(&min_buffer_lock_id);
  this->device_->unlock_buffer(&max_buffer_lock_id);
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Observe_out_gpu(size_t n,
                                                vptr<const void> data) {
  Observe_out_gpu(n, vptr<const MOtype>(data));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Observe_out_gpu(size_t n,
                                                vptr<const MOtype> data) {
  if (mode_ == PASSIVE) {
    return;
  }
  this->quant_mutex_.lock();
  if (!program_ready_) {
    this->GenerateKernels();
  }
  this->quant_mutex_.unlock();

  float scal = std::max(std::abs(max_out_), std::abs(min_out_))
                             / std::max(std::abs(flt_max_), std::abs(flt_min_));

  vector<size_t> local(1, 64);
  vector<size_t> group(1, (n - 1) / 64 + 1);

  int_tp min_buffer_lock_id = -1;
  int_tp max_buffer_lock_id = -1;

  shared_ptr<Blob<float> > min_blob =
      this->device_->template Buffer<float>(vector<int_tp>(1, group[1]),
                                            &min_buffer_lock_id);
  shared_ptr<Blob<float> > max_blob =
      this->device_->template Buffer<float>(vector<int_tp>(1, group[1]),
                                            &max_buffer_lock_id);

  vptr<float> min_gpu_data = min_blob->mutable_gpu_data();
  this->device_->template set<float>(n, type_max_val<float>(), min_gpu_data);
  vptr<float> max_gpu_data = max_blob->mutable_gpu_data();
  this->device_->template set<float>(n, type_min_val<float>(), max_gpu_data);

  shared_ptr<DeviceKernel> kernel =
                          this->program_->GetKernel("quantizer_observe_out");
  kernel->add_arg(&n);
  kernel->add_arg(&data);
  if (is_signed_integer_type<MItype>()) {
    kernel->add_arg(&scal);
  }
  kernel->add_arg(&min_gpu_data);
  kernel->add_arg(&max_gpu_data);
  kernel->Execute(group, local);

  const float* min_cpu_data = min_blob->cpu_data();
  const float* max_cpu_data = max_blob->cpu_data();

  this->quant_mutex_.lock();
  for (size_t i = 0; i < group[0]; ++i) {
    observed_max_ = std::max(static_cast<double>(min_cpu_data[i]),
                             observed_min_);
    observed_min_ = std::min(static_cast<double>(max_cpu_data[i]),
                             observed_max_);
  }
  this->quant_mutex_.unlock();

  this->device_->unlock_buffer(&min_buffer_lock_id);
  this->device_->unlock_buffer(&max_buffer_lock_id);
}

INSTANTIATE_CLASS_2T(Quantizer, PROTO_TYPES, PROTO_TYPES)

}  // namespace caffe
