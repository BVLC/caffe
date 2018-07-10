#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void PReLULayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                   "n", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                   "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                   "dim", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                   "in", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                   "out", KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                   "slope_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                   "div_factor", KERNEL_ARG_CONST));
    ss << this->device_program_->function("PReLUForward", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
    ss << "int_tp c = (index / dim) % channels / div_factor;" << std::endl;
    ss << "out[index] = in[index] > (Dtype)0 ? in[index] : in[index]"
       << " * slope_data[c];"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                   "n", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                   "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                   "dim", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                   "in_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                   "in_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                   "out_diff", KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                   "slope_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                   "div_factor", KERNEL_ARG_CONST));
    ss << this->device_program_->function("PReLUBackward", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
    ss << "int_tp c = (index / dim) % channels / div_factor;" << std::endl;
    ss << "out_diff[index] = in_diff[index]"
       << " * (((in_data[index] > (Dtype)0) ? (Dtype)1 : (Dtype)0)"
       << " + ((in_data[index] <= (Dtype)0) ? (Dtype)1 : (Dtype)0)"
       << " * slope_data[c]);"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                   "n", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                   "rows", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                   "rowPitch", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                   "in_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                   "in_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                   "out_diff", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("PReLUParamBackward", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
    ss << "out_diff[index] = in_diff[index] * in_data[index]"
       << " * ((in_data[index] <= (Dtype)0) ? (Dtype)1: (Dtype)0);"
       << std::endl;
    ss << "for (int k = 1; k < rows; k++) {" << std::endl;
    ss << "out_diff[index] += in_diff[index + k * rowPitch]"
       << " * in_data[index + k * rowPitch]"
       << " * ((in_data[index + k * rowPitch] <= (Dtype)0)"
       << " ? (Dtype)1 : (Dtype)0);" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}

template<typename Dtype, typename MItype, typename MOtype>
void PReLULayer<Dtype, MItype, MOtype>::Forward_gpu(
                                      const vector<Blob<MItype>*>& bottom,
                                      const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  const int_tp count = bottom[0]->count();
  const int_tp dim = bottom[0]->count(2);
  const int_tp channels = bottom[0]->shape(1);
  vptr<const Dtype> slope_data = this->blobs_[0]->gpu_data();
  const int_tp div_factor = channel_shared_ ? channels : 1;

  // For in-place computation
  if (top[0] == bottom[0]) {
    this->device_->template copy<Dtype>(count, bottom_data,
                                        bottom_memory_.mutable_gpu_data());
  }

  shared_ptr<DeviceKernel> kernel =
                               this->device_program_->GetKernel("PReLUForward");
  kernel->add_arg(&count);
  kernel->add_arg(&channels);
  kernel->add_arg(&dim);
  kernel->add_arg(&bottom_data);
  kernel->add_arg(&top_data);
  kernel->add_arg(&slope_data);
  kernel->add_arg(&div_factor);

  vector<size_t> work_size(1, count);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<typename Dtype, typename MItype, typename MOtype>
void PReLULayer<Dtype, MItype, MOtype>::Backward_gpu(
                                      const vector<Blob<MOtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<MItype>*>& bottom) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<const Dtype> top_diff = top[0]->gpu_diff();
  const int_tp count = bottom[0]->count();
  const int_tp dim = bottom[0]->count(2);
  const int_tp channels = bottom[0]->shape(1);

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computation), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    vptr<Dtype> slope_diff = this->blobs_[0]->mutable_gpu_diff();
    int_tp cdim = channels * dim;

    vptr<Dtype> backward_buff_diff = backward_buff_.mutable_gpu_diff();
    int_tp num = bottom[0]->shape(0);
    int_tp top_offset = top[0]->offset(1);

    shared_ptr<DeviceKernel> kernel =
                         this->device_program_->GetKernel("PReLUParamBackward");
    kernel->add_arg(&cdim);
    kernel->add_arg(&num);
    kernel->add_arg(&top_offset);
    kernel->add_arg(&top_diff);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&backward_buff_diff);

    vector<size_t> work_size(1, cdim);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);

    if (channel_shared_) {
      Dtype dsum;
      this->device_->template dot<Dtype>(channels * dim,
                     backward_buff_.gpu_diff(), multiplier_.gpu_data(), &dsum);
      this->device_->template add_scalar<Dtype>(this->blobs_[0]->count(),
                                                Dtype(dsum), slope_diff);
    } else {
      this->device_->template gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
                     backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
                     slope_diff);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
    vptr<const Dtype> slope_data = this->blobs_[0]->gpu_data();
    int_tp div_factor = channel_shared_ ? channels : 1;

    shared_ptr<DeviceKernel> kernel =
                              this->device_program_->GetKernel("PReLUBackward");
    kernel->add_arg(&count);
    kernel->add_arg(&channels);
    kernel->add_arg(&dim);
    kernel->add_arg(&top_diff);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&bottom_diff);
    kernel->add_arg(&slope_data);
    kernel->add_arg(&div_factor);

    vector<size_t> work_size(1, count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(PReLULayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PReLULayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PReLULayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(PReLULayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PReLULayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PReLULayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(PReLULayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PReLULayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(PReLULayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
