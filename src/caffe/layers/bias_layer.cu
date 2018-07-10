#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void BiasLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

  KernelArgs fw_args;
  fw_args.push_back(this->device_program_->template
                    create_kernel_arg<uint_tp>("n", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "in", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "bias", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "bias_dim", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "inner_dim", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "out", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("BiasForward", fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "const int_tp bias_index = (index / inner_dim) % bias_dim;"
     << std::endl;
  ss << "out[index] = in[index] + bias[bias_index];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}

template<typename Dtype, typename MItype, typename MOtype>
void BiasLayer<Dtype, MItype, MOtype>::Forward_gpu(
      const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
  const uint_tp count = top[0]->count();
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<const Dtype> bias_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();

  shared_ptr<DeviceKernel> kernel =
                                this->device_program_->GetKernel("BiasForward");
  kernel->add_arg(&count);
  kernel->add_arg(&bottom_data);
  kernel->add_arg(&bias_data);
  kernel->add_arg(&bias_dim_);
  kernel->add_arg(&inner_dim_);
  kernel->add_arg(&top_data);

  vector<size_t> work_size(1, count);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<typename Dtype, typename MItype, typename MOtype>
void BiasLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                    const vector<Blob<MOtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0] && bottom[0] != top[0]) {
    vptr<const Dtype> top_diff = top[0]->gpu_diff();
    vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
    this->device_->template copy<Dtype>(bottom[0]->count(),
                                        top_diff, bottom_diff);
  }
  // in-place, we don't need to do anything with the data diff
  const bool bias_param = (bottom.size() == 1);
  if ((!bias_param && propagate_down[1])
      || (bias_param && this->param_propagate_down_[0])) {
    vptr<const Dtype> top_diff = top[0]->gpu_diff();
    vptr<Dtype> bias_diff = (bias_param ? this->blobs_[0].get() : bottom[1])
        ->mutable_gpu_diff();
    bool accum = bias_param;

    for (int_tp n = 0; n < outer_dim_; ++n) {
      this->device_->gemv(CblasNoTrans, bias_dim_, inner_dim_, Dtype(1),
                          top_diff, bias_multiplier_.gpu_data(),
                          Dtype(accum), bias_diff);
      top_diff += dim_;
      accum = true;
    }
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(BiasLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BiasLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BiasLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(BiasLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BiasLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BiasLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(BiasLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BiasLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BiasLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
