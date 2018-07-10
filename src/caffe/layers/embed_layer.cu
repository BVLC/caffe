#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/embed_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void EmbedLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");
  ss << this->device_program_->atomics();

  KernelArgs fw_args;
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "nthreads", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "bottom_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "weight", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "M", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "N", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "K", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "top_data", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("EmbedForward", fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "top_index", "nthreads");
  ss << "const int_tp n = top_index / N;" << std::endl;
  ss << "const int_tp d = top_index % N;" << std::endl;
  ss << "const int_tp index = (int_tp)(bottom_data[n]);" << std::endl;
  ss << "const int_tp weight_index = index * N + d;" << std::endl;
  ss << "top_data[top_index] = weight[weight_index];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "nthreads", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "bottom_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "top_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "m", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "n", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "k", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "weight_diff", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("EmbedBackward", bw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "top_index", "nthreads");
  ss << "const int_tp b = top_index / n;" << std::endl;
  ss << "const int_tp d = top_index % n;" << std::endl;
  ss << "const int_tp index = (int_tp)(bottom_data[b]);" << std::endl;
  ss << "const int_tp weight_index = index * n + d;" << std::endl;
  ss << this->device_program_->template atomic_add<Dtype>(
      "weight_diff + weight_index", "top_diff[top_index]") << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}


template<typename Dtype, typename MItype, typename MOtype>
void EmbedLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                      const vector<Blob<MItype>*>& bottom,
                                      const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  vptr<const Dtype> weight = this->blobs_[0]->gpu_data();
  const int_tp count = top[0]->count();

  shared_ptr<DeviceKernel> kernel =
                               this->device_program_->GetKernel("EmbedForward");
  kernel->add_arg(&count);
  kernel->add_arg(&bottom_data);
  kernel->add_arg(&weight);
  kernel->add_arg(&M_);
  kernel->add_arg(&N_);
  kernel->add_arg(&K_);
  kernel->add_arg(&top_data);

  vector<size_t> work_size(1, count);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
  if (bias_term_) {
    this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
                                        Dtype(1), bias_multiplier_.gpu_data(),
                                        this->blobs_[1]->gpu_data(), Dtype(1),
                                        top_data);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void EmbedLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                      const vector<Blob<MOtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<MItype>*>& bottom) {
  CHECK(!propagate_down[0]) << "Can't backpropagate to EmbedLayer input.";
  if (this->param_propagate_down_[0]) {
    const int_tp top_count = top[0]->count();
    vptr<const Dtype> top_diff = top[0]->gpu_diff();
    vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
    vptr<Dtype> weight_diff = this->blobs_[0]->mutable_gpu_diff();

    shared_ptr<DeviceKernel> kernel =
                              this->device_program_->GetKernel("EmbedBackward");
    kernel->add_arg(&top_count);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&top_diff);
    kernel->add_arg(&M_);
    kernel->add_arg(&N_);
    kernel->add_arg(&K_);
    kernel->add_arg(&weight_diff);

    vector<size_t> work_size(1, top_count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    vptr<const Dtype> top_diff = top[0]->gpu_diff();
    vptr<Dtype> bias_diff = this->blobs_[1]->mutable_gpu_diff();
    this->device_->template gemv<Dtype>(CblasTrans, M_, N_, Dtype(1), top_diff,
        bias_multiplier_.gpu_data(), Dtype(1), bias_diff);
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(EmbedLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(EmbedLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(EmbedLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(EmbedLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(EmbedLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(EmbedLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(EmbedLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(EmbedLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(EmbedLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe

