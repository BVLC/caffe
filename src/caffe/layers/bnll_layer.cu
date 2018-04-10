#include <algorithm>
#include <vector>

#include "caffe/layers/bnll_layer.hpp"

namespace caffe {

const float kBNLL_THRESHOLD = 50.;


template<typename Dtype, typename MItype, typename MOtype>
void BNLLLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

  ss << "#define kBNLL_THRESHOLD " << kBNLL_THRESHOLD << std::endl;

  KernelArgs fw_args;
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "n", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "in", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "out", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("BNLLForward", fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "out[index] = in[index] > 0 ? ";
  ss << "in[index] + log(1. + exp(-in[index])) : log(1. + exp(in[index]));";
  ss << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "n", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "in_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "in_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "out_diff", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("BNLLBackward", bw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "Dtype expval = exp(min(in_data[index], Dtype(kBNLL_THRESHOLD)));"
     << std::endl;
  ss << "out_diff[index] = in_diff[index] * expval / (expval + 1.);"
     << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}


template<typename Dtype, typename MItype, typename MOtype>
void BNLLLayer<Dtype, MItype, MOtype>::Forward_gpu(
                            const vector<Blob<MItype>*>& bottom,
                            const vector<Blob<MOtype>*>& top) {
  vptr<const MItype> bottom_data = bottom[0]->gpu_data();
  vptr<MOtype> top_data = top[0]->mutable_gpu_data();
  const uint_tp count = bottom[0]->count();

  shared_ptr<DeviceKernel> kernel =
                                this->device_program_->GetKernel("BNLLForward");
  kernel->add_arg(&count);
  kernel->add_arg(&bottom_data);
  kernel->add_arg(&top_data);

  vector<size_t> work_size(1, count);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}


template<typename Dtype, typename MItype, typename MOtype>
void BNLLLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                   const vector<Blob<MOtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    vptr<const MItype> bottom_data = bottom[0]->gpu_data();
    vptr<const MOtype> top_diff = top[0]->gpu_diff();
    vptr<MItype> bottom_diff = bottom[0]->mutable_gpu_diff();
    const uint_tp count = bottom[0]->count();

    shared_ptr<DeviceKernel> kernel =
                               this->device_program_->GetKernel("BNLLBackward");
    kernel->add_arg(&count);
    kernel->add_arg(&top_diff);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&bottom_diff);

    vector<size_t> work_size(1, count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(BNLLLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BNLLLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BNLLLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(BNLLLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BNLLLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BNLLLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(BNLLLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BNLLLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BNLLLayer, Backward_gpu,
                                  (double), (double), (double));
}  // namespace caffe
