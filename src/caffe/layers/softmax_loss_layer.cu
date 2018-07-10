#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void SoftmaxWithLossLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");
  ss << this->device_program_->template helper_functions<Dtype>();

  KernelArgs fw_args;
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "nthreads", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "prob_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "label", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "loss", KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "num", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "dim", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "spatial_dim", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<bool>(
                    "has_ignore_label", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "ignore_label", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "counts", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("SoftmaxLossForwardGPU", fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
  ss << "const int_tp n = index / spatial_dim;" << std::endl;
  ss << "const int_tp s = index % spatial_dim;" << std::endl;
  ss << "const int_tp label_value = (int_tpc)(label[n * spatial_dim + s]);"
     << std::endl;
  ss << "if (has_ignore_label && label_value == ignore_label) {" << std::endl;
  ss << "loss[index] = (Dtype)0;" << std::endl;
  ss << "counts[index] = (Dtype)0;" << std::endl;
  ss << "} else {" << std::endl;
  ss << "loss[index] = -log("
     << "max((float)prob_data[n * dim + label_value * spatial_dim + s], "
     << "(float)FLT_MIN));" << std::endl;
  ss << "counts[index] = 1;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "nthreads", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "top", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "label", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "bottom_diff", KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "num", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "dim", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "spatial_dim", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<bool>(
                    "has_ignore_label", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "ignore_label", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "counts", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("SoftmaxLossBackwardGPU", bw_args);
  ss << "const int_tp channels = dim / spatial_dim;" << std::endl;
  ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
  ss << "const int_tp n = index / spatial_dim;" << std::endl;
  ss << "const int_tp s = index % spatial_dim;" << std::endl;
  ss << "const int_tp label_value = (int_tpc)(label[n * spatial_dim + s]);"
     << std::endl;
  ss << "if (has_ignore_label && label_value == ignore_label) {" << std::endl;
  ss << "for (int_tp c = 0; c < channels; ++c) {" << std::endl;
  ss << "bottom_diff[n * dim + c * spatial_dim + s] = 0;" << std::endl;
  ss << "}" << std::endl;
  ss << "counts[index] = 0;" << std::endl;
  ss << "} else {" << std::endl;
  ss << "bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;"
     << std::endl;
  ss << "counts[index] = 1;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}


template<typename Dtype, typename MItype, typename MOtype>
void SoftmaxWithLossLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  vptr<const Dtype> prob_data = prob_.gpu_data();
  vptr<const Dtype> label = bottom[1]->gpu_data();
  const int_tp dim = prob_.count() / outer_num_;
  const int_tp nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything, we use it here to avoid having
  // to allocate new GPU memory to accumulate intermediate results.
  vptr<Dtype> loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  vptr<Dtype> counts = prob_.mutable_gpu_diff();

  shared_ptr<DeviceKernel> kernel =
                      this->device_program_->GetKernel("SoftmaxLossForwardGPU");
  kernel->add_arg(&nthreads);
  kernel->add_arg(&prob_data);
  kernel->add_arg(&label);
  kernel->add_arg(&loss_data);
  kernel->add_arg(&outer_num_);
  kernel->add_arg(&dim);
  kernel->add_arg(&inner_num_);
  kernel->add_arg(&has_ignore_label_);
  kernel->add_arg(&ignore_label_);
  kernel->add_arg(&counts);

  vector<size_t> work_size(1, nthreads);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);

  Dtype loss;
  this->device_->template asum<Dtype>(nthreads, loss_data, &loss);
  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID
      && has_ignore_label_) {
    this->device_->template asum<Dtype>(nthreads, counts, &valid_count);
  }
  top[0]->mutable_cpu_data()[0] = loss
      / get_normalizer(normalization_, valid_count);
  if (top.size() >= 2) {
    top[1]->ShareData(prob_);
  }

  // Clear scratch memory to prevent interfering with backward (see #6202).
  this->device_->template set<Dtype>(bottom[0]->count(), Dtype(0),
                                     bottom[0]->mutable_gpu_diff());
}

template<typename Dtype, typename MItype, typename MOtype>
void SoftmaxWithLossLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) <<
        this->type() << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
    vptr<const Dtype> prob_data = prob_.gpu_data();
    vptr<const Dtype> top_data = top[0]->gpu_data();
    this->device_->template copy<Dtype>(prob_.count(), prob_data, bottom_diff);
    vptr<const Dtype> label = bottom[1]->gpu_data();
    const int_tp dim = prob_.count() / outer_num_;
    const int_tp nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    vptr<Dtype> counts = prob_.mutable_gpu_diff();

    shared_ptr<DeviceKernel> kernel =
                     this->device_program_->GetKernel("SoftmaxLossBackwardGPU");
    kernel->add_arg(&nthreads);
    kernel->add_arg(&top_data);
    kernel->add_arg(&label);
    kernel->add_arg(&bottom_diff);
    kernel->add_arg(&outer_num_);
    kernel->add_arg(&dim);
    kernel->add_arg(&inner_num_);
    kernel->add_arg(&has_ignore_label_);
    kernel->add_arg(&ignore_label_);
    kernel->add_arg(&counts);

    vector<size_t> work_size(1, nthreads);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);

    Dtype valid_count = -1;
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      this->device_->template asum<Dtype>(nthreads, counts, &valid_count);
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0] /
    get_normalizer(normalization_, valid_count);
    this->device_->template scal<Dtype>(prob_.count(), loss_weight,
                                        bottom_diff);
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxWithLossLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxWithLossLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxWithLossLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxWithLossLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxWithLossLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxWithLossLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxWithLossLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxWithLossLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SoftmaxWithLossLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
