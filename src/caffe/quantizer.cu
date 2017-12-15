#include "caffe/quantizer.hpp"

namespace caffe {

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::GenerateKernels() {
  this->program_ = this->device_->CreateProgram();
  stringstream ss;
  ss << this->program_->setup();
  ss << this->program_->define_type<MItype>(MItype);
  ss << this->program_->define_type<MOtype>(MOtype);

  {
    KernelArgs args;
    args.push_back(this->program_->template create_kernel_arg<uint_tp>("n",
                                                             KERNEL_ARG_CONST));
    args.push_back(this->program_->template create_kernel_arg<MItype>("in",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
    args.push_back(this->program_->template create_kernel_arg<MOtype>("out",
                                  KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM));
    if (fw_scale_before_cast()) {
      args.push_back(this->program_->template create_kernel_arg<MItype>("scal",
                                                             KERNEL_ARG_CONST));
    } else {
      args.push_back(this->program_->template create_kernel_arg<MOtype>("scal",
                                                             KERNEL_ARG_CONST));
    }
    ss << this->program_->function("quantizer_forward", args);
    ss << this->program_->kernel_loop("uint_tp", 'i', 'n');
    ss << "out[i] = " << fw_scale_term(0, "scal", "in[i]") << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->program_->template create_kernel_arg<uint_tp>("n",
                                                             KERNEL_ARG_CONST));
    args.push_back(this->program_->template create_kernel_arg<MOtype>("in",
               KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
    args.push_back(this->program_->template create_kernel_arg<MItype>("out",
                                  KERNEL_ARG_RESTRICT | KERNEL_ARG_GLOBAL_MEM));
    if (bw_scale_before_cast()) {
      args.push_back(this->program_->template create_kernel_arg<MOtype>("scal",
                                                             KERNEL_ARG_CONST));
    } else {
      args.push_back(this->program_->template create_kernel_arg<MItype>("scal",
                                                             KERNEL_ARG_CONST));
    }
    ss << this->program_->function("quantizer_backward", args);
    ss << this->program_->kernel_loop("uint_tp", 'i', 'n');
    ss << "out[i] = " << bw_scale_term(0, "scal", "in[i]") << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  this->program_->Compile(true, true);
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Forward_gpu(size_t n, vptr<const MItype> input,
                         vptr<MOtype> output) {
  if (!program_ready_) {
    this->GenerateKernels();
  }
  MItype scal_before = fw_scale_before_cast_val();
  MOtype scal_after = fw_scale_after_cast_val();

  shared_ptr<DeviceKernel> kernel =
                          this->device_program_->GetKernel("quantizer_forward");
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
void Quantizer<MItype, MOtype>::Backward_gpu(size_t n, vptr<const MOtype> input,
                         vptr<MItype> output) {
  if (!program_ready_) {
    this->GenerateKernels();
  }
  MOtype scal_before = bw_scale_before_cast_val();
  MItype scal_after = bw_scale_after_cast_val();

  shared_ptr<DeviceKernel> kernel =
                         this->device_program_->GetKernel("quantizer_backward");
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
void Quantizer<MItype, MOtype>::Forward_gpu(size_t n, vptr<const void> input,
                         vptr<void> output) {
  this->Forward_gpu(n, vptr<const MItype>(input), vptr<MOtype>(output));
}

template<typename MItype, typename MOtype>
void Quantizer<MItype, MOtype>::Backward_gpu(size_t n, vptr<const void> input,
                         vptr<void> output) {
  this->Backward_gpu(n, vptr<const MOtype>(input), vptr<MItype>(output));
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


INSTANTIATE_CLASS_2T(Quantizer, VARIANT_TYPES, VARIANT_TYPES)

}  // namespace caffe
