#include <vector>

#include "caffe/layers/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void ConcatLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

  KernelArgs args;
  args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "nthreads", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "in_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  args.push_back(this->device_program_->template create_kernel_arg<bool>(
                    "forward", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "num_concats", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "concat_size", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "top_concat_axis", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "bottom_concat_axis", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "offset_concat_axis", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "out_data", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("Concat", args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
  ss << "const int_tp total_concat_size = concat_size * bottom_concat_axis;"
     << std::endl;
  ss << "const int_tp concat_num = index / total_concat_size;" << std::endl;
  ss << "const int_tp concat_index = index % total_concat_size;" << std::endl;
  ss << "const int_tp top_index = concat_index"
     << " + (concat_num * top_concat_axis + offset_concat_axis) * concat_size;"
     << std::endl;
  ss << "if (forward) {" << std::endl;
  ss << "out_data[top_index] = in_data[index];" << std::endl;
  ss << "} else {" << std::endl;
  ss << "out_data[index] = in_data[top_index];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}

template<typename Dtype, typename MItype, typename MOtype>
void ConcatLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                            const vector<Blob<MItype>*>& bottom,
                                            const vector<Blob<MOtype>*>& top) {
  if (bottom.size() == 1) { return; }
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  int_tp offset_concat_axis = 0;
  const int_tp top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = true;

  shared_ptr<DeviceKernel> kernel = this->device_program_->GetKernel("Concat");

  for (int_tp i = 0; i < bottom.size(); ++i) {
    vptr<const Dtype> bottom_data = bottom[i]->gpu_data();
    const int_tp bottom_concat_axis = bottom[i]->shape(concat_axis_);
    const int_tp bottom_concat_size = bottom_concat_axis * concat_input_size_;
    const int_tp nthreads = bottom_concat_size * num_concats_;

    kernel->add_arg(&nthreads);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&kForward);
    kernel->add_arg(&num_concats_);
    kernel->add_arg(&concat_input_size_);
    kernel->add_arg(&top_concat_axis);
    kernel->add_arg(&bottom_concat_axis);
    kernel->add_arg(&offset_concat_axis);
    kernel->add_arg(&top_data);

    vector<size_t> work_size(1, nthreads);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);

    offset_concat_axis += bottom_concat_axis;
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void ConcatLayer<Dtype, MItype, MOtype>::Backward_gpu(
          const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
          const vector<Blob<MItype>*>& bottom) {
  if (bottom.size() == 1) { return; }
  vptr<const Dtype> top_diff = top[0]->gpu_diff();
  int_tp offset_concat_axis = 0;
  const int_tp top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = false;
  shared_ptr<DeviceKernel> kernel = this->device_program_->GetKernel("Concat");
  for (int_tp i = 0; i < bottom.size(); ++i) {
    const int_tp bottom_concat_axis = bottom[i]->shape(concat_axis_);
    if (propagate_down[i]) {
      vptr<Dtype> bottom_diff = bottom[i]->mutable_gpu_diff();
      const int_tp bottom_concat_axis = bottom[i]->shape(concat_axis_);
      const int_tp bottom_concat_size = bottom_concat_axis * concat_input_size_;
      const int_tp nthreads = bottom_concat_size * num_concats_;


      kernel->add_arg(&nthreads);
      kernel->add_arg(&top_diff);
      kernel->add_arg(&kForward);
      kernel->add_arg(&num_concats_);
      kernel->add_arg(&concat_input_size_);
      kernel->add_arg(&top_concat_axis);
      kernel->add_arg(&bottom_concat_axis);
      kernel->add_arg(&offset_concat_axis);
      kernel->add_arg(&bottom_diff);

      vector<size_t> work_size(1, nthreads);
      vector<size_t> group;
      vector<size_t> local;
      this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                 true);
      kernel->Execute(group, local);
    }
    offset_concat_axis += bottom_concat_axis;
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConcatLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConcatLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConcatLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConcatLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConcatLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConcatLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConcatLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConcatLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ConcatLayer, Backward_gpu,
                                  (double), (double), (double));
}  // namespace caffe
