#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void ReLULayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  typedef typename std::conditional<float_is_same<MItype>::value, MItype,
          typename std::conditional<sizeof(MItype) == 1, int16_t,
          typename std::conditional<sizeof(MItype) == 2, int32_t,
                                    int64_t>::type>::type>::type Difftype;
  typedef typename std::conditional<float_is_same<MItype>::value, MItype,
          typename std::conditional<sizeof(MItype) == 1, int32_t,
                                    int64_t>::type>::type Acctype;
  if (is_integer_type<MItype>()) {
    if (this->device_->template preferred_vector_width<int64_t>() > 0) {
      ss << this->device_program_->template define_vector_type<int64_t>(
          "Multtype", 0, 16);
    } else {
      ss << this->device_program_->template define_vector_type<int32_t>(
          "Multtype", 0, 16);
    }
  }

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");
  ss << this->device_program_->template define_type<Difftype>("Difftype");
  ss << this->device_program_->template define_type<Acctype>("Acctype");

  KernelArgs fw_args;
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "n", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "in", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "out", KERNEL_ARG_GLOBAL_MEM));
  if (is_float_type<Dtype>()) {
    fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "negative_slope", KERNEL_ARG_CONST));
  } else {
    fw_args.push_back(this->device_program_->template
                      create_kernel_arg<int8_t>("shift_bits",
                                                  KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template
                      create_kernel_arg<Difftype>("in_zero", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template
                      create_kernel_arg<Acctype>("mult", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template
                      create_kernel_arg<int8_t>("shift", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template
                      create_kernel_arg<Acctype>("out_zero", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template
                      create_kernel_arg<Acctype>("out_min", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template
                      create_kernel_arg<Acctype>("out_max", KERNEL_ARG_CONST));
  }

  ss << this->device_program_->function("ReLUForward", fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  if (is_float_type<Dtype>()) {
  ss << "out[index] = in[index] > (Dtype)0 ? in[index] : in[index]"
     << " * negative_slope;"
     << std::endl;
  } else {
    ss << "Difftype relu = max((Difftype)((Difftype)(in[index]) - "
       << "in_zero), (Difftype)0);" << std::endl;
    ss << "Acctype reg = (Acctype)(((Multtype)(relu) * "
       << "(Multtype)(mult)) / ((Multtype)1 << shift_bits));" << std::endl;
    ss << "if (shift >= 0) {" << std::endl;
    ss << "reg = reg >> shift;" << std::endl;
    ss << "} else {" << std::endl;
    ss << "reg = reg << -shift;" << std::endl;
    ss << "}" << std::endl;
    ss << "out[index] = (Dtype)(min(max(reg + out_zero, out_min), out_max));"
       << std::endl;
  }
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "n", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "in_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "in_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "out_diff", KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "negative_slope", KERNEL_ARG_CONST));
  ss << this->device_program_->function("ReLUBackward", bw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "out_diff[index] = in_diff[index]"
     << " * (((in_data[index] > (Dtype)0) ? (Dtype)1 : (Dtype)0)"
     << " + ((in_data[index] <= (Dtype)0) ? (Dtype)1 : (Dtype)0)"
     << " * negative_slope);"
     << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}

template<typename Dtype, typename MItype, typename MOtype>
void ReLULayer<Dtype, MItype, MOtype>::Forward_gpu(
                                     const vector<Blob<MItype>*>& bottom,
                                     const vector<Blob<MOtype>*>& top) {

  typedef typename std::conditional<float_is_same<MItype>::value, MItype,
          typename std::conditional<sizeof(MItype) == 1, int16_t,
          typename std::conditional<sizeof(MItype) == 2, int32_t,
                                    int64_t>::type>::type>::type Difftype;
  typedef typename std::conditional<float_is_same<MItype>::value, MItype,
          typename std::conditional<sizeof(MItype) == 1, int32_t,
                                    int64_t>::type>::type Acctype;

  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  const int_tp count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

  shared_ptr<DeviceKernel> kernel =
                                this->device_program_->GetKernel("ReLUForward");

  vector<size_t> work_size(1, count);
  vector<size_t> group;
  vector<size_t> local;

  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);

  kernel->add_arg(&count);
  kernel->add_arg(&bottom_data);
  kernel->add_arg(&top_data);
  if (is_float_type<Dtype>()) {
    kernel->add_arg(&negative_slope);
  } else {
    int8_t shift_bits =
        (this->device_->template preferred_vector_width<int64_t>() > 0
            ? 32 : 16) / sizeof(MItype) - 1;
    Acctype mult;
    int8_t shift;
    QuantizerValues bottom_qv = this->bottom_quants_[0]->out_quantizer_values();
    QuantizerValues top_qv = this->top_quants_[0]->in_quantizer_values();
    QuantizerBase::ScaleQuantVals<Acctype>(&bottom_qv, &top_qv,
                                           &mult, &shift, shift_bits);
    Difftype bottom_zero = bottom_qv.get_zero<Difftype>();
    Acctype top_zero = top_qv.get_zero<Acctype>();
    Acctype top_min = top_qv.get_min<Acctype>();
    Acctype top_max = top_qv.get_max<Acctype>();
    kernel->add_arg(&shift_bits);
    kernel->add_arg(&bottom_zero);
    kernel->add_arg(&mult);
    kernel->add_arg(&shift);
    kernel->add_arg(&top_zero);
    kernel->add_arg(&top_min);
    kernel->add_arg(&top_max);
  }
  kernel->Execute(group, local);
}


template<typename Dtype, typename MItype, typename MOtype>
void ReLULayer<Dtype, MItype, MOtype>::Backward_gpu(
                                     const vector<Blob<MOtype>*>& top,
                                     const vector<bool>& propagate_down,
                                     const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
    vptr<const Dtype> top_diff = top[0]->gpu_diff();
    vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
    const int_tp count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

    shared_ptr<DeviceKernel> kernel =
                               this->device_program_->GetKernel("ReLUBackward");
    kernel->add_arg(&count);
    kernel->add_arg(&top_diff);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&bottom_diff);
    kernel->add_arg(&negative_slope);

    vector<size_t> work_size(1, count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, GenerateProgram,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, GenerateProgram,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, GenerateProgram,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, GenerateProgram,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, GenerateProgram,
                                  (uint64_t), (uint64_t), (uint64_t));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, Forward_gpu,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, Forward_gpu,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, Forward_gpu,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, Forward_gpu,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, Forward_gpu,
                                  (uint64_t), (uint64_t), (uint64_t));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, Backward_gpu,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, Backward_gpu,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, Backward_gpu,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, Backward_gpu,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(ReLULayer, Backward_gpu,
                                  (uint64_t), (uint64_t), (uint64_t));
}  // namespace caffe
