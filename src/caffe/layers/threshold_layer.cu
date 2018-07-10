#include <vector>

#include "caffe/layers/threshold_layer.hpp"


namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void ThresholdLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

  KernelArgs args;
  args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "n", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "threshold", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "in", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "out", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("ThresholdForward", args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "out[index] = (Dtype)(in[index]) > threshold ?"
     << " MOtype(1) : MOtype(0);" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}

template<typename Dtype, typename MItype, typename MOtype>
void ThresholdLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                          const vector<Blob<MItype>*>& bottom,
                                          const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  const int_tp count = bottom[0]->count();

  shared_ptr<DeviceKernel> kernel =
                           this->device_program_->GetKernel("ThresholdForward");
  kernel->add_arg(&count);
  kernel->add_arg(&threshold_);
  kernel->add_arg(&bottom_data);
  kernel->add_arg(&top_data);

  vector<size_t> work_size(1, count);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ThresholdLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ThresholdLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ThresholdLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ThresholdLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ThresholdLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ThresholdLayer, Forward_gpu,
                                  (double), (double), (double));


}  // namespace caffe
