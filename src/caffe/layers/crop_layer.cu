#include <vector>

#include "caffe/layers/crop_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void CropLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

  KernelArgs fw_args;
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "nthreads", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "ndims", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "src_strides", KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "dst_strides", KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "offsets", KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "src", KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "dst", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("CropForward", fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
  ss << "int_tp dst_index = index;" << std::endl;
  ss << "int_tp src_index = 0;" << std::endl;
  ss << "for (int_tp i = 0; i < ndims; ++i) {" << std::endl;
  ss << "int_tp coord = dst_index / dst_strides[i];" << std::endl;
  ss << "dst_index -= coord * dst_strides[i];" << std::endl;
  ss << "src_index += src_strides[i] * (coord + offsets[i]);" << std::endl;
  ss << "}" << std::endl;
  ss << "dst[index] = src[src_index];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "nthreads", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "ndims", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "src_strides", KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "dst_strides", KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "offsets", KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "src", KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "dst", KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  ss << this->device_program_->function("CropBackward", bw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
  ss << "int_tp dst_index = index;" << std::endl;
  ss << "int_tp src_index = 0;" << std::endl;
  ss << "for (int_tp i = 0; i < ndims; ++i) {" << std::endl;
  ss << "int_tp coord = dst_index / dst_strides[i];" << std::endl;
  ss << "dst_index -= coord * dst_strides[i];" << std::endl;
  ss << "src_index += src_strides[i] * (coord + offsets[i]);" << std::endl;
  ss << "}" << std::endl;
  ss << "src[src_index] = dst[index];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}


template<typename Dtype, typename MItype, typename MOtype>
void CropLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                const vector<Blob<MItype>*>& bottom,
                                const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  uint_tp n = top[0]->count();
  int_tp num_axes = bottom[0]->num_axes();
  vptr<const int_tp> src_strides_data = src_strides_.gpu_data();
  vptr<const int_tp> dst_strides_data = dst_strides_.gpu_data();
  vptr<const int_tp> offsets_data = offsets.gpu_data();

  shared_ptr<DeviceKernel> kernel =
                                this->device_program_->GetKernel("CropForward");
  kernel->add_arg(&n);
  kernel->add_arg(&num_axes);
  kernel->add_arg(&src_strides_data);
  kernel->add_arg(&dst_strides_data);
  kernel->add_arg(&offsets_data);
  kernel->add_arg(&bottom_data);
  kernel->add_arg(&top_data);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<typename Dtype, typename MItype, typename MOtype>
void CropLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                    const vector<Blob<MOtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Blob<MItype>*>& bottom) {
  vptr<const Dtype> top_diff = top[0]->gpu_diff();
  vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
  int_tp n = top[0]->count();
  int_tp num_axes = bottom[0]->num_axes();
  vptr<const int_tp> src_strides_data = src_strides_.gpu_data();
  vptr<const int_tp> dst_strides_data = dst_strides_.gpu_data();
  vptr<const int_tp> offsets_data = offsets.gpu_data();

  shared_ptr<DeviceKernel> kernel =
                               this->device_program_->GetKernel("CropBackward");

  if (propagate_down[0]) {
    this->device_->set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);

    kernel->add_arg(&n);
    kernel->add_arg(&num_axes);
    kernel->add_arg(&src_strides_data);
    kernel->add_arg(&dst_strides_data);
    kernel->add_arg(&offsets_data);
    kernel->add_arg(&bottom_diff);
    kernel->add_arg(&top_diff);

    vector<size_t> work_size(1, n);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(CropLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(CropLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(CropLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(CropLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(CropLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(CropLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(CropLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(CropLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(CropLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
