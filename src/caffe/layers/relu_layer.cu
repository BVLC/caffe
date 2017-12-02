#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void ReLULayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

  KernelArgs fw_args;
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "n", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "in", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "out", KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "negative_slope", KERNEL_ARG_CONST));
  ss << this->device_program_->function("ReLUForward", fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;"
     << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "n", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "in_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "in_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "out_diff", KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "negative_slope", KERNEL_ARG_CONST));
  ss << this->device_program_->function("ReLUBackward", bw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "out_diff[index] = in_diff[index]"
     << " * ((in_data[index] > 0) + (in_data[index] <= 0) * negative_slope);"
     << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}

template<typename Dtype, typename MItype, typename MOtype>
void ReLULayer<Dtype, MItype, MOtype>::Forward_gpu(
                                     const vector<Blob<MItype>*>& bottom,
                                     const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  const int_tp count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

  shared_ptr<DeviceKernel> kernel =
                                this->device_program_->GetKernel("ReLUForward");
  kernel->add_arg(&count);
  kernel->add_arg(&bottom_data);
  kernel->add_arg(&top_data);
  kernel->add_arg(&negative_slope);

  vector<size_t> work_size(1, count);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}



template<typename Dtype, typename MItype, typename MOtype>
void ReLULayer<Dtype, MItype, MOtype>::Backward_gpu(
                                     const vector<Blob<MOtype>*>& top,
                                     const vector<bool>& propagate_down,
                                     const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
    vptr<const Dtype> top_diff = top[0]->gpu_diff();
    vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
    const int_tp count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

    shared_ptr<DeviceKernel> kernel =
                               this->device_program_->GetKernel("ReLUBackward");
    kernel->add_arg(&count);
    kernel->add_arg(&top_diff);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&bottom_diff);
    kernel->add_arg(&negative_slope);

    vector<size_t> work_size(1, count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }
}

INSTANTIATE_CLASS_3T_GUARDED(ReLULayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(ReLULayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(ReLULayer, (double), (double), (double));

}  // namespace caffe
