#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void DropoutLayer<Dtype, MItype, MOtype>::GenerateProgram() {
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
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "mask", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "threshold", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "scale", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "out", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("DropoutForward", fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "out[index] = in[index] * (Dtype)(mask[index] > threshold) * scale;"
     << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "n", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "in_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "mask", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "threshold", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "scale", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "out_diff", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("DropoutBackward", bw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "out_diff[index] = in_diff[index] * scale *"
     << " (Dtype)(mask[index] > threshold);"
     << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}


template<typename Dtype, typename MItype, typename MOtype>
void DropoutLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                      const vector<Blob<MItype>*>& bottom,
                                      const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  const int_tp count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    vptr<uint_tp> mask = rand_vec_.mutable_gpu_data();
    this->device_->rng_uniform(count, mask);

    shared_ptr<DeviceKernel> kernel =
                             this->device_program_->GetKernel("DropoutForward");
    kernel->add_arg(&count);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&mask);
    kernel->add_arg(&uint_thres_);
    kernel->add_arg(&scale_);
    kernel->add_arg(&top_data);

    vector<size_t> work_size(1, count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  } else {
    this->device_->template copy<Dtype>(count, bottom_data, top_data);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void DropoutLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                       const vector<Blob<MOtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    vptr<const Dtype> top_diff = top[0]->gpu_diff();
    vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      vptr<const uint_tp> mask = rand_vec_.gpu_data();
      const int_tp count = bottom[0]->count();
      shared_ptr<DeviceKernel> kernel =
                            this->device_program_->GetKernel("DropoutBackward");
      kernel->add_arg(&count);
      kernel->add_arg(&top_diff);
      kernel->add_arg(&mask);
      kernel->add_arg(&uint_thres_);
      kernel->add_arg(&scale_);
      kernel->add_arg(&bottom_diff);

      vector<size_t> work_size(1, count);
      vector<size_t> group;
      vector<size_t> local;
      this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                 true);
    } else {
      this->device_->template copy<Dtype>(top[0]->count(), top_diff,
                                          bottom_diff);
    }
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, GenerateProgram,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, GenerateProgram,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, GenerateProgram,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, GenerateProgram,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, GenerateProgram,
                                  (uint64_t), (uint64_t), (uint64_t));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, Forward_gpu,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, Forward_gpu,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, Forward_gpu,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, Forward_gpu,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, Forward_gpu,
                                  (uint64_t), (uint64_t), (uint64_t));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, Backward_gpu,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, Backward_gpu,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, Backward_gpu,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, Backward_gpu,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutLayer, Backward_gpu,
                                  (uint64_t), (uint64_t), (uint64_t));
}  // namespace caffe
