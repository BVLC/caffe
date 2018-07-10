#include <cfloat>
#include <vector>

#include "caffe/layers/scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void ScaleLayer<Dtype, MItype, MOtype>::GenerateProgram() {
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
                    "scale", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "scale_dim", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "inner_dim", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "out", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("ScaleForward", fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "const int_tp scale_index = (index / inner_dim) % scale_dim;"
     << std::endl;
  ss << "out[index] = in[index] * scale[scale_index];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "n", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "in", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "scale", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "bias", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "scale_dim", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "inner_dim", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "out", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("ScaleBiasForward", bw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "const int_tp scale_index = (index / inner_dim) % scale_dim;"
     << std::endl;
  ss << "out[index] = in[index] * scale[scale_index] + bias[scale_index];"
     << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}

template<typename Dtype, typename MItype, typename MOtype>
void ScaleLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                      const vector<Blob<MItype>*>& bottom,
                                      const vector<Blob<MOtype>*>& top) {
  const int_tp count = top[0]->count();
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();

  if (bottom[0] == top[0]) {
    this->device_->template copy<Dtype>(bottom[0]->count(),
                   bottom[0]->gpu_data(), temp_.mutable_gpu_data());
  }
  vptr<const Dtype> scale_data = (
      (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  if (bias_layer_) {
    vptr<const Dtype> bias_data = this->blobs_[bias_param_id_]->gpu_data();

    shared_ptr<DeviceKernel> kernel =
                           this->device_program_->GetKernel("ScaleBiasForward");
    kernel->add_arg(&count);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&scale_data);
    kernel->add_arg(&bias_data);
    kernel->add_arg(&scale_dim_);
    kernel->add_arg(&inner_dim_);
    kernel->add_arg(&top_data);

    vector<size_t> work_size(1, count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  } else {
    shared_ptr<DeviceKernel> kernel =
                               this->device_program_->GetKernel("ScaleForward");
    kernel->add_arg(&count);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&scale_data);
    kernel->add_arg(&scale_dim_);
    kernel->add_arg(&inner_dim_);
    kernel->add_arg(&top_data);

    vector<size_t> work_size(1, count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void ScaleLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                      const vector<Blob<MOtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<MItype>*>& bottom) {
  if (bias_layer_
      && this->param_propagate_down_[this->param_propagate_down_.size() - 1]) {
    bias_layer_->Backward(top, bias_propagate_down_, bias_bottom_vec_);
  }
  const bool scale_param = (bottom.size() == 1);
  Blob<Dtype>* scale = scale_param ? this->blobs_[0].get() : bottom[1];

  if ((!scale_param && propagate_down[1])
      || (scale_param && this->param_propagate_down_[0])) {
    vptr<const Dtype> top_diff = top[0]->gpu_diff();
    const bool in_place = (bottom[0] == top[0]);
    vptr<const Dtype> bottom_data = (in_place ? &temp_ : bottom[0])->gpu_data();
    const bool is_eltwise = (bottom[0]->count() == scale->count());
    vptr<Dtype> product = (
        is_eltwise ?
            scale->mutable_gpu_diff() :
            (in_place ?
                temp_.mutable_gpu_data() : bottom[0]->mutable_gpu_diff()));
    this->device_->mul(top[0]->count(), top_diff, bottom_data, product);
    if (!is_eltwise) {
      vptr<Dtype> sum_result;
      if (inner_dim_ == 1) {
        sum_result = product;
      } else if (sum_result_.count() == 1) {
        vptr<const Dtype> sum_mult = sum_multiplier_.gpu_data();
        Dtype* scale_diff = scale->mutable_cpu_diff();
        if (scale_param) {
          Dtype result;
          this->device_->template dot<Dtype>(inner_dim_, product, sum_mult,
                                             &result);
          *scale_diff += result;
        } else {
          this->device_->template dot<Dtype>(inner_dim_, product, sum_mult,
                                             scale_diff);
        }
      } else {
        vptr<const Dtype> sum_mult = sum_multiplier_.gpu_data();
        sum_result =
            (outer_dim_ == 1) ?
                scale->mutable_gpu_diff() : sum_result_.mutable_gpu_data();
        this->device_->template gemv<Dtype>(CblasNoTrans, sum_result_.count(),
                       inner_dim_, Dtype(1), product, sum_mult, Dtype(0),
                       sum_result);
      }
      if (outer_dim_ != 1) {
        vptr<const Dtype> sum_mult = sum_multiplier_.gpu_data();
        if (scale_dim_ == 1) {
          Dtype* scale_diff = scale->mutable_cpu_diff();
          if (scale_param) {
            Dtype result;
            this->device_->template dot<Dtype>(outer_dim_, sum_mult,
                                               sum_result, &result);
            *scale_diff += result;
          } else {
            this->device_->template dot<Dtype>(outer_dim_, sum_mult,
                                               sum_result, scale_diff);
          }
        } else {
          vptr<Dtype> scale_diff = scale->mutable_gpu_diff();
          this->device_->template gemv<Dtype>(CblasTrans, outer_dim_,
                         scale_dim_, Dtype(1), sum_result, sum_mult,
                         Dtype(scale_param), scale_diff);
        }
      }
    }
  }
  if (propagate_down[0]) {
    const int_tp count = top[0]->count();
    vptr<const Dtype> top_diff = top[0]->gpu_diff();
    vptr<const Dtype> scale_data = scale->gpu_data();
    vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();

    shared_ptr<DeviceKernel> kernel =
                               this->device_program_->GetKernel("ScaleForward");
    kernel->add_arg(&count);
    kernel->add_arg(&top_diff);
    kernel->add_arg(&scale_data);
    kernel->add_arg(&scale_dim_);
    kernel->add_arg(&inner_dim_);
    kernel->add_arg(&bottom_diff);

    vector<size_t> work_size(1, count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ScaleLayer, GenerateProgram,
                                 (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ScaleLayer, GenerateProgram,
                                 (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ScaleLayer, GenerateProgram,
                                 (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ScaleLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ScaleLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ScaleLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ScaleLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ScaleLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ScaleLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
