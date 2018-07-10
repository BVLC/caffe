#include <cmath>
#include <vector>

#include "caffe/layers/swish_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void SwishLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

  KernelArgs fw_args;
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "n", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "in", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "out", KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "beta", KERNEL_ARG_CONST));
  ss << this->device_program_->function("SwishForward", fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "const Dtype sigmoid_in = (Dtype)0.5"
     << " * (Dtype)tanh((Dtype)0.5 * beta * in[index]) + (Dtype)0.5;"
     << std::endl;
  ss << "out[index] = in[index] * sigmoid_in;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "n", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "in_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "in_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "out_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "out_diff", KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "beta", KERNEL_ARG_CONST));
  ss << this->device_program_->function("SwishBackward", bw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "const Dtype sigmoid_out = (Dtype)0.5"
     << " * (Dtype)tanh((Dtype)0.5 * beta * out_data[index]) + (Dtype)0.5;"
     << std::endl;
  ss << "const Dtype swish_x = in_data[index];" << std::endl;
  ss << "out_diff[index] = in_diff[index] * (beta * swish_x"
     << " + sigmoid_out * ((Dtype)1.0 - beta * swish_x));" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}

template<typename Dtype, typename MItype, typename MOtype>
void SwishLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  vptr<const MItype> bottom_data = bottom[0]->gpu_data();
  vptr<MOtype> top_data = top[0]->mutable_gpu_data();
  const int_tp count = bottom[0]->count();
  Dtype beta = this->layer_param_.swish_param().beta();

  shared_ptr<DeviceKernel> kernel =
                               this->device_program_->GetKernel("SwishForward");
  kernel->add_arg(&count);
  kernel->add_arg(&bottom_data);
  kernel->add_arg(&top_data);
  kernel->add_arg(&beta);

  vector<size_t> work_size(1, count);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}


template<typename Dtype, typename MItype, typename MOtype>
void SwishLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    vptr<const MOtype> top_data = top[0]->gpu_data();
    vptr<const MOtype> top_diff = top[0]->gpu_diff();
    vptr<const MItype> bottom_data = bottom[0]->gpu_data();
    vptr<MItype> bottom_diff = bottom[0]->mutable_gpu_diff();
    const int_tp count = bottom[0]->count();
    Dtype beta = this->layer_param_.swish_param().beta();

    shared_ptr<DeviceKernel> kernel =
                              this->device_program_->GetKernel("SwishBackward");
    kernel->add_arg(&count);
    kernel->add_arg(&top_data);
    kernel->add_arg(&top_diff);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&bottom_diff);
    kernel->add_arg(&beta);

    vector<size_t> work_size(1, count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(SwishLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SwishLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SwishLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(SwishLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SwishLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SwishLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(SwishLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SwishLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SwishLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
