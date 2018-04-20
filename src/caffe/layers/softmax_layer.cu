#include <algorithm>
#include <cfloat>
#include <vector>

#ifdef USE_CUDA
#include "thrust/device_vector.h"
#endif

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void SoftmaxLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");
  ss << this->device_program_->template helper_functions<Dtype>();

#ifdef USE_HALF
  if (std::is_same<MItype, half_fp>::value) {
    ss << "#define DTYPE_MAX HALF_MAX" << std::endl;
    ss << "#define DTYPE_MIN HALF_MIN" << std::endl;
  } else if (std::is_same<MItype, float>::value
        || std::is_same<MItype, double>::value) {
#endif
    ss << "#define DTYPE_MAX FLT_MAX" << std::endl;
    ss << "#define DTYPE_MIN FLT_MIN" << std::endl;
#ifdef USE_HALF
  } else {
    ss << "#define DTYPE_MAX " << type_max_val<MItype>() << std::endl;
    ss << "#define DTYPE_MIN " << 0 << std::endl;
  }
#endif

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "spatial_dim", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "out", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("kernel_channel_max", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index",
                                             "num * spatial_dim");
    ss << "int_tp n = index / spatial_dim;" << std::endl;
    ss << "int_tp s = index % spatial_dim;" << std::endl;
    ss << "Dtype maxval = -DTYPE_MAX;" << std::endl;
    ss << "for (int_tp c = 0; c < channels; ++c) {" << std::endl;
    ss << "maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);"
       << std::endl;
    ss << "}" << std::endl;
    ss << "out[index] = maxval;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "count", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "spatial_dim", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "channel_max", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "data", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("kernel_channel_subtract", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "count");
    ss << "int_tp n = index / channels / spatial_dim;" << std::endl;
    ss << "int_tp s = index % spatial_dim;" << std::endl;
    ss << "data[index] -= channel_max[n * spatial_dim + s];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "count", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "out", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("kernel_exp", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "count");
    ss << "out[index] = exp(data[index]);" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "spatial_dim", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "channel_sum", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("kernel_channel_sum", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index",
                                             "num * spatial_dim");
    ss << "int_tp n = index / spatial_dim;" << std::endl;
    ss << "int_tp s = index % spatial_dim;" << std::endl;
    ss << "Dtype sum = 0;" << std::endl;
    ss << "for (int_tp c = 0; c < channels; ++c) {" << std::endl;
    ss << "sum += data[(n * channels + c) * spatial_dim + s];" << std::endl;
    ss << "}" << std::endl;
    ss << "channel_sum[index] = sum;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "count", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "spatial_dim", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "channel_sum", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "data", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("kernel_channel_div", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "count");
    ss << "int_tp n = index / channels / spatial_dim;" << std::endl;
    ss << "int_tp s = index % spatial_dim;" << std::endl;
    ss << "data[index] /= channel_sum[n * spatial_dim + s];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "spatial_dim", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "data_1", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "data_2", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "channel_dot", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("kernel_channel_dot", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index",
                                             "num * spatial_dim");
    ss << "int_tp n = index / spatial_dim;" << std::endl;
    ss << "int_tp s = index % spatial_dim;" << std::endl;
    ss << "Dtype dot = 0;" << std::endl;
    ss << "for (int_tp c = 0; c < channels; ++c) {" << std::endl;
    ss << "dot += (data_1[(n * channels + c) * spatial_dim + s]"
       << " * data_2[(n * channels + c) * spatial_dim + s]);" << std::endl;
    ss << "}" << std::endl;
    ss << "channel_dot[index] = dot;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}


template<typename Dtype, typename MItype, typename MOtype>
void SoftmaxLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                        const vector<Blob<MItype>*>& bottom,
                                        const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  vptr<Dtype> scale_data = scale_.mutable_gpu_data();
  int_tp count = bottom[0]->count();
  int_tp channels = top[0]->shape(softmax_axis_);

  // CUDA backend code
  this->device_->template copy<Dtype>(count, bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.

  // compute max
  {
    shared_ptr<DeviceKernel> kernel =
                         this->device_program_->GetKernel("kernel_channel_max");
    kernel->add_arg(&outer_num_);
    kernel->add_arg(&channels);
    kernel->add_arg(&inner_num_);
    kernel->add_arg(&top_data);
    kernel->add_arg(&scale_data);

    vector<size_t> work_size(1, outer_num_ * inner_num_);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }

  // subtract
  {
    shared_ptr<DeviceKernel> kernel =
                    this->device_program_->GetKernel("kernel_channel_subtract");
    kernel->add_arg(&count);
    kernel->add_arg(&outer_num_);
    kernel->add_arg(&channels);
    kernel->add_arg(&inner_num_);
    kernel->add_arg(&scale_data);
    kernel->add_arg(&top_data);

    vector<size_t> work_size(1, count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }

  // exponentiate
  {
    shared_ptr<DeviceKernel> kernel =
                                 this->device_program_->GetKernel("kernel_exp");
    kernel->add_arg(&count);
    kernel->add_arg(&top_data);
    kernel->add_arg(&top_data);

    vector<size_t> work_size(1, count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }

  // sum after exp
  {
    shared_ptr<DeviceKernel> kernel =
                         this->device_program_->GetKernel("kernel_channel_sum");
    kernel->add_arg(&outer_num_);
    kernel->add_arg(&channels);
    kernel->add_arg(&inner_num_);
    kernel->add_arg(&top_data);
    kernel->add_arg(&scale_data);

    vector<size_t> work_size(1, outer_num_ * inner_num_);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }

  // divide
  {
    shared_ptr<DeviceKernel> kernel =
                         this->device_program_->GetKernel("kernel_channel_div");
    kernel->add_arg(&count);
    kernel->add_arg(&outer_num_);
    kernel->add_arg(&channels);
    kernel->add_arg(&inner_num_);
    kernel->add_arg(&scale_data);
    kernel->add_arg(&top_data);

    vector<size_t> work_size(1, count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void SoftmaxLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                        const vector<Blob<MOtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<MItype>*>& bottom) {
  vptr<const Dtype> top_diff = top[0]->gpu_diff();
  vptr<const Dtype> top_data = top[0]->gpu_data();
  vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
  vptr<Dtype> scale_data = scale_.mutable_gpu_data();
  int_tp count = top[0]->count();
  int_tp channels = top[0]->shape(softmax_axis_);

  this->device_->template copy<Dtype>(top[0]->count(), top_diff, bottom_diff);
  // Compute inner1d(top_diff, top_data) and
  // subtract them from the bottom diff.
  {
    shared_ptr<DeviceKernel> kernel =
                         this->device_program_->GetKernel("kernel_channel_dot");
    kernel->add_arg(&outer_num_);
    kernel->add_arg(&channels);
    kernel->add_arg(&inner_num_);
    kernel->add_arg(&top_diff);
    kernel->add_arg(&top_data);
    kernel->add_arg(&scale_data);

    vector<size_t> work_size(1, outer_num_ * inner_num_);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }

  {
    shared_ptr<DeviceKernel> kernel =
                    this->device_program_->GetKernel("kernel_channel_subtract");
    kernel->add_arg(&count);
    kernel->add_arg(&outer_num_);
    kernel->add_arg(&channels);
    kernel->add_arg(&inner_num_);
    kernel->add_arg(&scale_data);
    kernel->add_arg(&bottom_diff);

    vector<size_t> work_size(1, count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }

  // Elementwise multiplication
  this->device_->template mul<Dtype>(top[0]->count(), bottom_diff,
                                     top_data, bottom_diff);
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, GenerateProgram,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, GenerateProgram,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, GenerateProgram,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, GenerateProgram,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, GenerateProgram,
                                  (uint64_t), (uint64_t), (uint64_t));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, Forward_gpu,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, Forward_gpu,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, Forward_gpu,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, Forward_gpu,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, Forward_gpu,
                                  (uint64_t), (uint64_t), (uint64_t));


INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, Backward_gpu,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, Backward_gpu,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, Backward_gpu,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, Backward_gpu,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxLayer, Backward_gpu,
                                  (uint64_t), (uint64_t), (uint64_t));
}  // namespace caffe
