#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void SigmoidCrossEntropyLossLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

  KernelArgs fw_args;
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "count", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "input_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "target", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "loss", KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<bool>(
                    "has_ignore_label", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "ignore_label", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "counts", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("SigmoidCrossEntropyLossForwardGPU",
                                        fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "i", "count");
  ss << "const int_tp target_value = (int_tpc)(target[i]);"
     << std::endl;
  ss << "if (has_ignore_label && target_value == ignore_label) {" << std::endl;
  ss << "loss[i] = (MItype)0;" << std::endl;
  ss << "counts[i] = (MItype)0;" << std::endl;
  ss << "} else {" << std::endl;
  ss << "loss[i] = input_data[i] * (target[i] -"
     << " ((input_data[i] >= (MItype)0) ? (MItype)1 : (MItype)0 )) -"
     << " (MItype)log((MItype)1 +"
     << " (MItype)exp(input_data[i] - (MItype)2 * input_data[i] *"
     << " ((input_data[i] >= (MItype)0) ? (MItype)1 : (MItype)0)));"
     << std::endl;
  ss << "counts[i] = (MItype)1;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "count", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "ignore_label", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "target", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "diff", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("SigmoidCrossEntropyLossIgnoreDiffGPU",
                                        bw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "i", "count");
  ss << "const int_tp target_value = (int_tpc)(target[i]);"
     << std::endl;
  ss << "if (target_value == ignore_label) {" << std::endl;
  ss << "diff[i] = (MItype)0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}

template<typename Dtype, typename MItype, typename MOtype>
void SigmoidCrossEntropyLossLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int_tp count = bottom[0]->count();
  const int_tp num = bottom[0]->num();
  // Stable version of loss computation from input data
  vptr<const MItype> input_data = bottom[0]->gpu_data();
  vptr<const MItype> target = bottom[1]->gpu_data();
  // Since this memory is not used for anything, we use it here to avoid having
  // to allocate new GPU memory to accumulate intermediate results.
  vptr<MItype> loss_data = bottom[0]->mutable_gpu_diff();

  MItype loss;
  vptr<MItype> count_data = bottom[1]->mutable_gpu_diff();
  MItype valid_count;

  shared_ptr<DeviceKernel> kernel =
          this->device_program_->GetKernel("SigmoidCrossEntropyLossForwardGPU");
  kernel->add_arg(&count);
  kernel->add_arg(&input_data);
  kernel->add_arg(&target);
  kernel->add_arg(&loss_data);
  kernel->add_arg(&has_ignore_label_);
  kernel->add_arg(&ignore_label_);
  kernel->add_arg(&count_data);

  vector<size_t> work_size(1, count);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);

  // Only launch another CUDA kernel if we actually need the valid count.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    this->device_->template asum<Dtype>(count, count_data, &valid_count);
  } else {
    valid_count = count;
  }
  this->device_->template asum<Dtype>(count, loss_data, &loss);
  normalizer_ = get_normalizer(normalization_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;

  // Clear scratch memory to prevent interfering with backward (see #6202).
  this->device_->template set<Dtype>(bottom[0]->count(),
                                     Dtype(0), bottom[0]->mutable_gpu_diff());
  this->device_->template set<Dtype>(bottom[1]->count(), Dtype(0),
                                     bottom[1]->mutable_gpu_diff());
}

template<typename Dtype, typename MItype, typename MOtype>
void SigmoidCrossEntropyLossLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL)<< this->type()
    << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int_tp count = bottom[0]->count();
    vptr<const Dtype> sigmoid_output_data = sigmoid_output_->gpu_data();
    vptr<const Dtype> target = bottom[1]->gpu_data();
    vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();

    // First, compute the diff
    this->device_->template copy<Dtype>(count, sigmoid_output_data,
                                        bottom_diff);
    this->device_->template axpy<Dtype>(count, Dtype(-1), target, bottom_diff);
    // Zero out gradient of ignored targets.
    if (has_ignore_label_) {

      shared_ptr<DeviceKernel> kernel =
       this->device_program_->GetKernel("SigmoidCrossEntropyLossIgnoreDiffGPU");
      kernel->add_arg(&count);
      kernel->add_arg(&ignore_label_);
      kernel->add_arg(&target);
      kernel->add_arg(&bottom_diff);

      vector<size_t> work_size(1, count);
      vector<size_t> group;
      vector<size_t> local;
      this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                 true);
      kernel->Execute(group, local);
    }
    // Scale down gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    this->device_->template scal<Dtype>(count, loss_weight, bottom_diff);
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(SigmoidCrossEntropyLossLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SigmoidCrossEntropyLossLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SigmoidCrossEntropyLossLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(SigmoidCrossEntropyLossLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SigmoidCrossEntropyLossLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SigmoidCrossEntropyLossLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(SigmoidCrossEntropyLossLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SigmoidCrossEntropyLossLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SigmoidCrossEntropyLossLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
