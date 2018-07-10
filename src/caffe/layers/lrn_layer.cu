#include <vector>

#include "caffe/layers/lrn_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void LRNLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");
  ss << this->device_program_->template helper_functions<Dtype>();

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "in", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "size", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "alpha_over_size", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "k", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "scale", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("LRNFillScale", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    // find out the local offset
    ss << "const int_tp w = index % width;" << std::endl;
    ss << "const int_tp h = (index / width) % height;" << std::endl;
    ss << "const int_tp n = index / width / height;" << std::endl;
    ss << "const int_tp offset = (n * channels * height + h) * width + w;"
       << std::endl;
    ss << "const int_tp step = height * width;" << std::endl;
    ss << this->device_program_->global_ptr("const Dtype", "in_off")
       << " = in + offset;" << std::endl;
    ss << this->device_program_->global_ptr("Dtype", "scale_off")
       << " = scale + offset;" << std::endl;
    ss << "int_tp head = 0;" << std::endl;
    ss << "const int_tp pre_pad = (size - 1) / 2;" << std::endl;
    ss << "const int_tp post_pad = size - pre_pad - 1;" << std::endl;
    ss << "Dtype accum_scale = 0;" << std::endl;
    // fill the scale at [n, :, h, w]
    // accumulate values
    ss << "while (head < post_pad && head < channels) {" << std::endl;
    ss << "accum_scale += in_off[head * step] * in_off[head * step];"
       << std::endl;
    ss << "++head;" << std::endl;
    ss << "}" << std::endl;
    // both add and subtract
    ss << "while (head < channels) {" << std::endl;
    ss << "accum_scale += in_off[head * step] * in_off[head * step];"
       << std::endl;
    ss << "if (head - size >= 0) {" << std::endl;
    ss << "accum_scale -= in_off[(head - size) * step]"
       << " * in_off[(head - size) * step];" << std::endl;
    ss << "}" << std::endl;
    ss << "scale_off[(head - post_pad) * step] = k + accum_scale"
       << " * alpha_over_size;" << std::endl;
    ss << "++head;" << std::endl;
    ss << "}" << std::endl;
    // subtract only
    ss << "while (head < channels + post_pad) {" << std::endl;
    ss << "if (head - size >= 0) {" << std::endl;
    ss << "accum_scale -= in_off[(head - size) * step]"
       << " * in_off[(head - size) * step];" << std::endl;
    ss << "}" << std::endl;
    ss << "scale_off[(head - post_pad) * step] = k + accum_scale"
       << " * alpha_over_size;" << std::endl;
    ss << "++head;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  // TODO: check if it would be faster to just put it into the previous kernel.
  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "in", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "scale", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "negative_beta", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "out", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("LRNComputeOutput", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    ss << "out[index] = in[index] * pow((Dtype)(scale[index]), "
       << "(Dtype)(negative_beta));"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                      "bottom_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                      "top_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "scale", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                      "top_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "channels", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "height", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "width", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "size", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "negative_beta", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "cache_ratio", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                      "bottom_diff", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("LRNComputeDiff", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");

    // find out the local offset
    ss << "const int_tp w = index % width;" << std::endl;
    ss << "const int_tp h = (index / width) % height;" << std::endl;
    ss << "const int_tp n = index / width / height;" << std::endl;
    ss << "const int_tp offset = (n * channels * height + h) * width + w;"
       << std::endl;
    ss << "const int_tp step = height * width;" << std::endl;
    ss << this->device_program_->global_ptr("const Dtype", "bottom_off")
       << " = bottom_data + offset;" << std::endl;
    ss << this->device_program_->global_ptr("const Dtype", "top_off")
       << " = top_data + offset;" << std::endl;
    ss << this->device_program_->global_ptr("const Dtype", "scale_off")
       << " = scale + offset;" << std::endl;
    ss << this->device_program_->global_ptr("const Dtype", "top_diff_off")
       << " = top_diff + offset;" << std::endl;
    ss << this->device_program_->global_ptr("Dtype", "bottom_diff_off")
       << " = bottom_diff + offset;" << std::endl;
    ss << "int_tp head = 0;" << std::endl;
    ss << "const int_tp pre_pad = size - (size + 1) / 2;" << std::endl;
    ss << "const int_tp post_pad = size - pre_pad - 1;" << std::endl;
    ss << "Dtype accum_ratio = 0;" << std::endl;
    // accumulate values
    ss << "while (head < post_pad && head < channels) {" << std::endl;
    ss << "accum_ratio += top_diff_off[head * step] * top_off[head * step]"
       << " / scale_off[head * step];" << std::endl;
    ss << "++head;" << std::endl;
    ss << "}" << std::endl;
    // both add and subtract
    ss << "while (head < channels) {" << std::endl;
    ss << "accum_ratio += top_diff_off[head * step] * top_off[head * step]"
       << " / scale_off[head * step];" << std::endl;
    ss << "if (head - size >= 0) {" << std::endl;
    ss << "accum_ratio -= top_diff_off[(head - size) * step]"
       << " * top_off[(head - size) * step] / scale_off[(head - size) * step];"
       << std::endl;
    ss << "}" << std::endl;
    ss << "bottom_diff_off[(head - post_pad) * step]"
       << " = top_diff_off[(head - post_pad)"
       << " * step] * pow((Dtype)(scale_off[(head - post_pad) * step]), "
       << "(Dtype)(negative_beta))"
       << " - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;"
       << std::endl;
    ss << "++head;" << std::endl;
    ss << "}" << std::endl;
    // subtract only
    ss << "while (head < channels + post_pad) {" << std::endl;
    ss << "if (head - size >= 0) {" << std::endl;
    ss << "accum_ratio -= top_diff_off[(head - size) * step]"
       << " * top_off[(head - size) * step] / scale_off[(head - size) * step];"
       << std::endl;
    ss << "}" << std::endl;
    ss << "bottom_diff_off[(head - post_pad) * step]"
       << " = top_diff_off[(head - post_pad)"
       << " * step] * pow((Dtype)(scale_off[(head - post_pad) * step]), "
       << "(Dtype)(negative_beta))"
       << " - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;"
       << std::endl;
    ss << "++head;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}

template<typename Dtype, typename MItype, typename MOtype>
void LRNLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                    const vector<Blob<MItype>*>& bottom,
                                    const vector<Blob<MOtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
    case LRNParameter_NormRegion_ACROSS_CHANNELS:
      CrossChannelForward_gpu(bottom, top);
      break;
    case LRNParameter_NormRegion_WITHIN_CHANNEL:
      WithinChannelForward(bottom, top);
      break;
    default:
      LOG(FATAL)<< "Unknown normalization region.";
  }
}


template<typename Dtype, typename MItype, typename MOtype>
void LRNLayer<Dtype, MItype, MOtype>::CrossChannelForward_gpu(
                                    const vector<Blob<MItype>*>& bottom,
                                    const vector<Blob<MOtype>*>& top) {
  // First, compute scale
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  vptr<Dtype> scale_data = scale_.mutable_gpu_data();
  Dtype neg_beta = -beta_;

  // We will launch one kernel for each pixel location, and have the kernel
  // go through all the channels.
  int_tp n_threads = num_ * height_ * width_;
  {
    shared_ptr<DeviceKernel> kernel =
                               this->device_program_->GetKernel("LRNFillScale");
    kernel->add_arg(&n_threads);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&num_);
    kernel->add_arg(&channels_);
    kernel->add_arg(&height_);
    kernel->add_arg(&width_);
    kernel->add_arg(&size_);
    Dtype alpha_size = alpha_ / size_;
    kernel->add_arg(&alpha_size);
    kernel->add_arg(&k_);
    kernel->add_arg(&scale_data);

    vector<size_t> work_size(1, n_threads);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }

  n_threads = bottom[0]->count();
  {
    shared_ptr<DeviceKernel> kernel =
                           this->device_program_->GetKernel("LRNComputeOutput");
    kernel->add_arg(&n_threads);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&scale_data);
    kernel->add_arg(&neg_beta);
    kernel->add_arg(&top_data);

    vector<size_t> work_size(1, n_threads);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void LRNLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                   const vector<Blob<MOtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<MItype>*>& bottom) {
switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelBackward_gpu(top, propagate_down, bottom);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelBackward(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL)<< "Unknown normalization region.";
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void LRNLayer<Dtype, MItype, MOtype>::CrossChannelBackward_gpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  int_tp n_threads = num_ * height_ * width_;
  Dtype neg_beta = -beta_;
  Dtype cache_ratio = Dtype(2. * alpha_ * beta_ / size_);
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<const Dtype> top_data = top[0]->gpu_data();
  vptr<const Dtype> scale_data = scale_.gpu_data();
  vptr<const Dtype> top_diff = top[0]->gpu_diff();
  vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();

  shared_ptr<DeviceKernel> kernel =
                             this->device_program_->GetKernel("LRNComputeDiff");
  kernel->add_arg(&n_threads);
  kernel->add_arg(&bottom_data);
  kernel->add_arg(&top_data);
  kernel->add_arg(&scale_data);
  kernel->add_arg(&top_diff);
  kernel->add_arg(&num_);
  kernel->add_arg(&channels_);
  kernel->add_arg(&height_);
  kernel->add_arg(&width_);
  kernel->add_arg(&size_);
  kernel->add_arg(&neg_beta);
  kernel->add_arg(&cache_ratio);
  kernel->add_arg(&bottom_diff);

  vector<size_t> work_size(1, n_threads);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(LRNLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LRNLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LRNLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(LRNLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LRNLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LRNLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(LRNLayer, CrossChannelForward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LRNLayer, CrossChannelForward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LRNLayer, CrossChannelForward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(LRNLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LRNLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LRNLayer, Backward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(LRNLayer, CrossChannelBackward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LRNLayer, CrossChannelBackward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LRNLayer, CrossChannelBackward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
