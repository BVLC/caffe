#include <algorithm>
#include <vector>

#include "caffe/layers/contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void ContrastiveLossLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

  KernelArgs args;
  args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "count", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "channels", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "margin", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<bool>(
                    "legacy_version", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "alpha", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "y", KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "diff", KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "dist_sq", KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "bottom_diff", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("CLLBackward", args);
  ss << this->device_program_->kernel_loop("uint_tp", "i", "count");
  // the num index, to access Y and dist_sq
  ss << "uint_tp n = i / channels;" << std::endl;
  ss << "if ((int_tp)(y[n])) {" << std::endl;  // similar pairs
  ss << "bottom_diff[i] = alpha * diff[i];" << std::endl;
  ss << "} else {" << std::endl;  // dissimilar pairs
  ss << "Dtype mdist = (Dtype)0;" << std::endl;
  ss << "Dtype beta = (Dtype)0;" << std::endl;
  ss << "if (legacy_version) {" << std::endl;
  ss << "mdist = (margin - dist_sq[n]);" << std::endl;
  ss << "beta = -alpha;" << std::endl;
  ss << "} else {" << std::endl;
  ss << "Dtype dist = sqrt(dist_sq[n]);" << std::endl;
  ss << "mdist = (margin - dist);" << std::endl;
  ss << "beta = -alpha * mdist / (dist + (Dtype)(1e-4)) * diff[i];" << std::endl;
  ss << "}" << std::endl;
  ss << "if (mdist > (Dtype)0) {" << std::endl;
  ss << "bottom_diff[i] = beta;" << std::endl;
  ss << "} else {" << std::endl;
  ss << "bottom_diff[i] = (MItype)0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}


template<typename Dtype, typename MItype, typename MOtype>
void ContrastiveLossLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const bool legacy_version = this->layer_param_.contrastive_loss_param()
      .legacy_version();

  const int_tp count = bottom[0]->count();

  this->device_->template sub<Dtype>(count, bottom[0]->gpu_data(),  // a
                                     bottom[1]->gpu_data(),  // b
                                     diff_.mutable_gpu_data());  // a_i-b_i
  // a_i-b_i
  this->device_->template powx<Dtype>(count, diff_.mutable_gpu_data(),
                         Dtype(2), diff_sq_.mutable_gpu_data());  // (a_i-b_i)^2
  this->device_->template gemv<Dtype>(CblasNoTrans, bottom[0]->shape(0),
                               bottom[0]->shape(1),
                               Dtype(1.0),
                               diff_sq_.gpu_data(),  // (a_i-b_i)^2
                               summer_vec_.gpu_data(), Dtype(0.0),
                               // \Sum (a_i-b_i)^2
                               dist_sq_.mutable_gpu_data());

  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  Dtype loss(0.0);
  for (int_tp i = 0; i < bottom[0]->shape(0); ++i) {
    if (static_cast<int_tp>(bottom[2]->cpu_data()[i])) {  // similar pairs
      loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
      if (legacy_version) {
        loss += fmax(margin - dist_sq_.cpu_data()[i], Dtype(0.0));
      } else {
        Dtype dist = fmax(margin - (Dtype) sqrt(dist_sq_.cpu_data()[i]),
                              Dtype(0.0));
        loss += dist * dist;
      }
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->shape(0)) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}


template<typename Dtype, typename MItype, typename MOtype>
void ContrastiveLossLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  const bool legacy_version = this->layer_param_.contrastive_loss_param()
      .legacy_version();

  shared_ptr<DeviceKernel> kernel =
                                this->device_program_->GetKernel("CLLBackward");

  for (int_tp i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const int_tp count = bottom[0]->count();
      const int_tp channels = bottom[0]->shape(1);
      Dtype margin = this->layer_param_.contrastive_loss_param().margin();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0]
          / static_cast<Dtype>(bottom[0]->shape(0));

      vptr<const Dtype> bottom_data = bottom[2]->gpu_data();
      vptr<const Dtype> diff_data = diff_.gpu_data();
      vptr<const Dtype> dist_sq_data = diff_sq_.gpu_data();
      vptr<Dtype> bottom_diff = bottom[i]->mutable_gpu_diff();

      kernel->add_arg(&count);
      kernel->add_arg(&channels);
      kernel->add_arg(&margin);
      kernel->add_arg(&legacy_version);
      kernel->add_arg(&alpha);
      kernel->add_arg(&bottom_data);  // pair similarity 0 or 1
      // the cached eltwise difference between a and b
      kernel->add_arg(&diff_data);
      // the cached square distance between a and b
      kernel->add_arg(&dist_sq_data);
      kernel->add_arg(&bottom_diff);

      vector<size_t> work_size(1, count);
      vector<size_t> group;
      vector<size_t> local;
      this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                 true);
      kernel->Execute(group, local);
    }
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ContrastiveLossLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ContrastiveLossLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ContrastiveLossLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ContrastiveLossLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ContrastiveLossLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ContrastiveLossLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ContrastiveLossLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ContrastiveLossLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ContrastiveLossLayer, Backward_gpu,
                                  (double), (double), (double));
}  // namespace caffe
