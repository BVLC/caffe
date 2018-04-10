#include <vector>

#include "caffe/layers/slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void SliceLayer<Dtype, MItype, MOtype>::GenerateProgram() {
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
                    "num_slices", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "slice_size", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "bottom_slice_axis", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "top_slice_axis", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "offset_slice_axis", KERNEL_ARG_CONST));
  args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "out_data", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("Slice", args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
  ss << "const int_tp total_slice_size = slice_size * top_slice_axis;"
     << std::endl;
  ss << "const int_tp slice_num = index / total_slice_size;" << std::endl;
  ss << "const int_tp slice_index = index % total_slice_size;" << std::endl;
  ss << "const int_tp bottom_index = slice_index"
     << " + (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;"
     << std::endl;
  ss << "if (forward) {" << std::endl;
  ss << "out_data[index] = in_data[bottom_index];" << std::endl;
  ss << "} else {" << std::endl;
  ss << "out_data[bottom_index] = in_data[index];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}


template<typename Dtype, typename MItype, typename MOtype>
void SliceLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                      const vector<Blob<MItype>*>& bottom,
                                      const vector<Blob<MOtype>*>& top) {
  if (top.size() == 1) { return; }
  int_tp offset_slice_axis = 0;
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  const int_tp bottom_slice_axis = bottom[0]->shape(slice_axis_);
  const bool kForward = true;
  for (int_tp i = 0; i < top.size(); ++i) {
    vptr<Dtype> top_data = top[i]->mutable_gpu_data();
    const int_tp top_slice_axis = top[i]->shape(slice_axis_);
    const int_tp top_slice_size = top_slice_axis * slice_size_;
    const int_tp nthreads = top_slice_size * num_slices_;

    shared_ptr<DeviceKernel> kernel = this->device_program_->GetKernel("Slice");
    kernel->add_arg(&nthreads);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&kForward);
    kernel->add_arg(&num_slices_);
    kernel->add_arg(&slice_size_);
    kernel->add_arg(&bottom_slice_axis);
    kernel->add_arg(&top_slice_axis);
    kernel->add_arg(&offset_slice_axis);
    kernel->add_arg(&top_data);

    vector<size_t> work_size(1, nthreads);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);

    offset_slice_axis += top_slice_axis;
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void SliceLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                      const vector<Blob<MOtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<MItype>*>& bottom) {
  if (!propagate_down[0] || top.size() == 1) { return; }
  int_tp offset_slice_axis = 0;
  vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
  const int_tp bottom_slice_axis = bottom[0]->shape(slice_axis_);
  const bool kForward = false;
  for (int_tp i = 0; i < top.size(); ++i) {
    vptr<const Dtype> top_diff = top[i]->gpu_diff();
    const int_tp top_slice_axis = top[i]->shape(slice_axis_);
    const int_tp top_slice_size = top_slice_axis * slice_size_;
    const int_tp nthreads = top_slice_size * num_slices_;

    shared_ptr<DeviceKernel> kernel = this->device_program_->GetKernel("Slice");
    kernel->add_arg(&nthreads);
    kernel->add_arg(&top_diff);
    kernel->add_arg(&kForward);
    kernel->add_arg(&num_slices_);
    kernel->add_arg(&slice_size_);
    kernel->add_arg(&bottom_slice_axis);
    kernel->add_arg(&top_slice_axis);
    kernel->add_arg(&offset_slice_axis);
    kernel->add_arg(&bottom_diff);

    vector<size_t> work_size(1, nthreads);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);

    offset_slice_axis += top_slice_axis;
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(SliceLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SliceLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SliceLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(SliceLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SliceLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SliceLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(SliceLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SliceLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(SliceLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
