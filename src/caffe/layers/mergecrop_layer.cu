#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/mergecrop_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void MergeCropLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

  KernelArgs fw_args;
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "nthreads", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "dims", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "bottom_a", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<bool>(
                    "forward_a", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "bottom_b", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<bool>(
                    "forward_b", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "top", KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "num", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "channels_a", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "channels_b", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "shape_a", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "shape_b", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("MergeCropForward", fw_args);
  ss << "int_tp pad[6];" << std::endl;
  ss << "int_tp tmp_idx[6]; " << std::endl;
  ss << "int_tp size_a = 1;" << std::endl;
  ss << "int_tp size_b = 1;" << std::endl;
  ss << "for (int_tp i = 0; i < dims; ++i) {" << std::endl;
  ss << "pad[i] = (shape_b[i] - shape_a[i]) / 2;" << std::endl;
  ss << "size_a *= shape_a[i];" << std::endl;
  ss << "size_b *= shape_b[i];" << std::endl;
  ss << "}" << std::endl;
  ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
  if (op_ == MergeCropParameter_MergeOp_STACK) {
    ss << "int_tp batch_id = index / ((channels_a + channels_b) * size_a);"
       << std::endl;
    ss << "int_tp bottom_id = ((index - batch_id * (channels_a + channels_b)"
       << " * size_a) / (channels_a * size_a)) % 2;" << std::endl;
    ss << "int_tp counter = index;" << std::endl;
    ss << "for (int_tp i = dims - 1; i >= 0; --i) {" << std::endl;
    ss << "tmp_idx[i] = counter % shape_a[i];" << std::endl;
    ss << "counter /= shape_a[i];" << std::endl;
    ss << "}" << std::endl;
    ss << "if (bottom_id == 0) {" << std::endl;
    ss << "int_tp channel_id = (index / size_a) % channels_a;" << std::endl;
    ss << "int_tp aidx = batch_id * channels_a + channel_id;" << std::endl;
    ss << "for (int_tp i = 0; i < dims; ++i) {" << std::endl;
    ss << "aidx *= shape_a[i];" << std::endl;
    ss << "aidx += tmp_idx[i];" << std::endl;
    ss << "}" << std::endl;
    ss << "top[index] = forward_a ? bottom_a[aidx] : 0;" << std::endl;
    ss << "} else {" << std::endl;
    ss << "int_tp channel_id = (index / size_a) % channels_b;" << std::endl;
    ss << "int_tp bidx = (batch_id * channels_b + channel_id) * size_b;"
       << std::endl;
    ss << "int_tp btemp = 1;" << std::endl;
    ss << "for (int_tp i = dims - 1; i >= 0; --i) {" << std::endl;
    ss << "bidx += btemp * (tmp_idx[i] + pad[i]);" << std::endl;
    ss << "btemp *= shape_b[i];" << std::endl;
    ss << "}" << std::endl;
    ss << "top[index] = forward_b ? bottom_b[bidx] : 0;" << std::endl;
    ss << "}" << std::endl;
  } else {
    ss << "int_tp batch_id = index / (channels_a * size_a);" << std::endl;
    ss << "int_tp counter = index;" << std::endl;
    ss << "for (int_tp i = dims - 1; i >= 0; --i) {" << std::endl;
    ss << "tmp_idx[i] = counter % shape_a[i];" << std::endl;
    ss << "counter /= shape_a[i];" << std::endl;
    ss << "}" << std::endl;

    ss << "top[index] = 0;" << std::endl;
    ss << "int_tp channel_id = (index / size_a) % channels_a;" << std::endl;
    ss << "int_tp aidx = batch_id * channels_a + channel_id;" << std::endl;
    ss << "for (int_tp i = 0; i < dims; ++i) {" << std::endl;
    ss << "aidx *= shape_a[i];" << std::endl;
    ss << "aidx += tmp_idx[i];" << std::endl;
    ss << "}" << std::endl;
    ss << "top[index] = forward_a ? top[index] + bottom_a[aidx] : top[index];"
       << std::endl;
    ss << "int_tp bidx = (batch_id * channels_a + channel_id) * size_b;"
       << std::endl;
    ss << "int_tp btemp = 1;" << std::endl;
    ss << "for (int_tp i = dims - 1; i >= 0; --i) {" << std::endl;
    ss << "bidx += btemp * (tmp_idx[i] + pad[i]);" << std::endl;
    ss << "btemp *= shape_b[i];" << std::endl;
    ss << "}" << std::endl;
    ss << "top[index] = forward_b ? top[index] + bottom_b[bidx] : top[index];"
       << std::endl;
  }
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "nthreads", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "dims", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "bottom_a", KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<bool>(
                    "backward_a", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "bottom_b", KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<bool>(
                    "backward_b", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "top", KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "num", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "channels_a", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "channels_b", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "shape_a", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "shape_b", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("MergeCropBackward", bw_args);
  ss << "int_tp pad[6];" << std::endl;
  ss << "int_tp tmp_idx[6];" << std::endl;
  ss << "int_tp size_a = 1;" << std::endl;
  ss << "int_tp size_b = 1;" << std::endl;

  ss << "for (int_tp i = 0; i < dims; ++i) {" << std::endl;
  ss << "pad[i] = (shape_b[i] - shape_a[i]) / 2;" << std::endl;
  ss << "size_a *= shape_a[i];" << std::endl;
  ss << "size_b *= shape_b[i];" << std::endl;
  ss << "}" << std::endl;

  ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
  if (op_ == MergeCropParameter_MergeOp_STACK) {
    ss << "int_tp batch_id = index / ((channels_a + channels_b) * size_a);"
       << std::endl;
    ss << "int_tp bottom_id = ((index - batch_id * (channels_a + channels_b)"
       << " * size_a) / (channels_a * size_a)) % 2;" << std::endl;
    ss << "int_tp counter = index;" << std::endl;
    ss << "for (int_tp i = dims - 1; i >= 0; --i) {" << std::endl;
    ss << "tmp_idx[i] = counter % shape_a[i];" << std::endl;
    ss << "counter /= shape_a[i];" << std::endl;
    ss << "}" << std::endl;

    ss << "if (bottom_id == 0) {" << std::endl;
    ss << "int_tp channel_id = (index / size_a) % channels_a;" << std::endl;
    ss << "int_tp aidx = batch_id * channels_a + channel_id;" << std::endl;
    ss << "for (int_tp i = 0; i < dims; ++i) {" << std::endl;
    ss << "aidx *= shape_a[i];" << std::endl;
    ss << "aidx += tmp_idx[i];" << std::endl;
    ss << "}" << std::endl;
    ss << "bottom_a[aidx] = backward_a ? top[index] : 0;" << std::endl;
    ss << "} else {" << std::endl;
    ss << "int_tp channel_id = (index / size_a) % channels_b;" << std::endl;
    ss << "int_tp bidx = (batch_id * channels_b + channel_id) * size_b;"
       << std::endl;
    ss << "int_tp btemp = 1;" << std::endl;
    ss << "for (int_tp i = dims - 1; i >= 0; --i) {" << std::endl;
    ss << "bidx += btemp * (tmp_idx[i] + pad[i]);" << std::endl;
    ss << "btemp *= shape_b[i];" << std::endl;
    ss << "}" << std::endl;
    ss << "bottom_b[bidx] = backward_b ? top[index] : 0;" << std::endl;
    ss << "}" << std::endl;
  } else {
    ss << "int_tp batch_id = index / (channels_a * size_a);" << std::endl;
    ss << "int_tp counter = index;" << std::endl;
    ss << "for (int_tp i = dims - 1; i >= 0; --i) {" << std::endl;
    ss << "tmp_idx[i] = counter % shape_a[i];" << std::endl;
    ss << "counter /= shape_a[i];" << std::endl;
    ss << "}" << std::endl;
    ss << "int_tp channel_id = (index / size_a) % channels_a;" << std::endl;
    ss << "int_tp aidx = batch_id * channels_a + channel_id;" << std::endl;
    ss << "for (int_tp i = 0; i < dims; ++i) {" << std::endl;
    ss << "aidx *= shape_a[i];" << std::endl;
    ss << "aidx += tmp_idx[i];" << std::endl;
    ss << "}" << std::endl;
    ss << "bottom_a[aidx] = backward_a ? top[index] : 0;" << std::endl;
    ss << "int_tp bidx = (batch_id * channels_a + channel_id) * size_b;"
       << std::endl;
    ss << "int_tp btemp = 1;" << std::endl;
    ss << "for (int_tp i = dims - 1; i >= 0; --i) {" << std::endl;
    ss << "bidx += btemp * (tmp_idx[i] + pad[i]);" << std::endl;
    ss << "btemp *= shape_b[i];" << std::endl;
    ss << "}" << std::endl;
    ss << "bottom_b[bidx] = backward_b ? top[index] : 0;" << std::endl;
  }
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}

template<typename Dtype, typename MItype, typename MOtype>
void MergeCropLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                          const vector<Blob<MItype>*>& bottom,
                                          const vector<Blob<MOtype>*>& top) {
  int_tp count = top[0]->count();

  vptr<const Dtype> bottom_data_a = bottom[0]->gpu_data();
  vptr<const Dtype> bottom_data_b = bottom[1]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();

  int_tp num = bottom[0]->shape(0);
  int_tp spatial_dims = bottom[0]->shape().size() - 2;

  // All channels of both inputs are copied
  int_tp channels_a = bottom[0]->shape(1);
  int_tp channels_b = bottom[1]->shape(1);

  vptr<const int_tp> shape_a_data = shape_a_.gpu_data();
  vptr<const int_tp> shape_b_data = shape_b_.gpu_data();

  shared_ptr<DeviceKernel> kernel =
                           this->device_program_->GetKernel("MergeCropForward");

  kernel->add_arg(&count);
  kernel->add_arg(&spatial_dims);
  kernel->add_arg(&bottom_data_a);
  bool fw_0 = forward_[0];
  kernel->add_arg(&fw_0);
  kernel->add_arg(&bottom_data_b);
  bool fw_1 = forward_[1];
  kernel->add_arg(&fw_1);
  kernel->add_arg(&top_data);
  kernel->add_arg(&num);
  kernel->add_arg(&channels_a);
  kernel->add_arg(&channels_b);
  kernel->add_arg(&shape_a_data);
  kernel->add_arg(&shape_b_data);

  vector<size_t> work_size(1, count);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<typename Dtype, typename MItype, typename MOtype>
void MergeCropLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                          const vector<Blob<MOtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<MItype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  int_tp count = top[0]->count();

  vptr<Dtype> bottom_diff_a = bottom[0]->mutable_gpu_diff();
  vptr<Dtype> bottom_diff_b = bottom[1]->mutable_gpu_diff();
  vptr<const Dtype> top_diff = top[0]->gpu_diff();

  int_tp num = bottom[0]->shape(0);
  int_tp spatial_dims = bottom[0]->shape().size() - 2;

  // All channels of both inputs are copied
  int_tp channels_a = bottom[0]->shape(1);
  int_tp channels_b = bottom[1]->shape(1);

  vptr<const int_tp> shape_a_data = shape_a_.gpu_data();
  vptr<const int_tp> shape_b_data = shape_b_.gpu_data();

  shared_ptr<DeviceKernel> kernel =
                          this->device_program_->GetKernel("MergeCropBackward");

  kernel->add_arg(&count);
  kernel->add_arg(&spatial_dims);
  kernel->add_arg(&bottom_diff_a);
  bool bw_0 = backward_[0];
  kernel->add_arg(&bw_0);
  kernel->add_arg(&bottom_diff_b);
  bool bw_1 = backward_[1];
  kernel->add_arg(&bw_1);
  kernel->add_arg(&top_diff);
  kernel->add_arg(&num);
  kernel->add_arg(&channels_a);
  kernel->add_arg(&channels_b);
  kernel->add_arg(&shape_a_data);
  kernel->add_arg(&shape_b_data);

  vector<size_t> work_size(1, count);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(MergeCropLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MergeCropLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MergeCropLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(MergeCropLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MergeCropLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MergeCropLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(MergeCropLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MergeCropLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(MergeCropLayer, Backward_gpu,
                                  (double), (double), (double));
}  // namespace caffe
