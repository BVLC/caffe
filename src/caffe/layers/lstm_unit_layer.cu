#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/lstm_layer.hpp"

namespace caffe {

template<typename Dtype>
string sigmoid(string name) {
  // Dtype(1) / (Dtype(1) + exp(-x));
  stringstream ss;
  ss << safe_type_name<Dtype>() << "(1) / "
     << "(" << safe_type_name<Dtype>() << "(1) + exp(-(" << name << ")))";
  return ss.str();
}

template<typename Dtype>
string tanh(string name) {
  // Dtype(2) * sigmoid(Dtype(2) * x) - Dtype(1);
  stringstream ss;
  ss << safe_type_name<Dtype>() << "(2) * "
     << sigmoid<Dtype>(safe_type_name<Dtype>() + "(2) * (" + name + ")")
     << " - " << safe_type_name<Dtype>() << "(1)";
  return ss.str();
}


template<typename Dtype, typename MItype, typename MOtype>
void LSTMUnitLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dim", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "x", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "x_acts", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("LSTMActsForward", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    ss << "const int_tp x_dim = 4 * dim;" << std::endl;
    ss << "const int_tp d = index % x_dim;" << std::endl;
    ss << "if (d < 3 * dim) {" << std::endl;
    ss << "x_acts[index] = " << sigmoid<Dtype>("x[index]") << std::endl;
    ss << "} else {" << std::endl;
    ss << "x_acts[index] = " << tanh<Dtype>("x[index]") << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dim", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "c_prev", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "x", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "cont", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "c", KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "h", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("LSTMUnitForward", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    ss << "const int_tp n = index / dim;" << std::endl;
    ss << "const int_tp d = index % dim;" << std::endl;
    ss << this->device_program_->global_ptr("const Dtype", "x_offset")
       << " = x + 4 * dim * n;" << std::endl;
    ss << "const Dtype i = x_offset[d];" << std::endl;
    ss << "const Dtype f = x_offset[1 * dim + d];" << std::endl;
    ss << "const Dtype o = x_offset[2 * dim + d];" << std::endl;
    ss << "const Dtype g = x_offset[3 * dim + d];" << std::endl;
    ss << "const Dtype c_prev_t = c_prev[index];" << std::endl;
    ss << "const Dtype c_t = cont[n] * f * c_prev_t + i * g;" << std::endl;
    ss << "c[index] = c_t;" << std::endl;
    ss << "const Dtype tanh_c = " << tanh<Dtype>("c_t") << ";" << std::endl;
    ss << "h[index] = o * tanh_c;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs args;
    args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                      "nthreads", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dim", KERNEL_ARG_CONST));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "x_acts", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "x_acts_diff", KERNEL_ARG_GLOBAL_MEM));
    args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                      "x_diff", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("LSTMActsBackward", args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
    ss << "const int_tp x_dim = 4 * dim;" << std::endl;
    ss << "const int_tp d = index % x_dim;" << std::endl;
    ss << "const Dtype x_act = x_acts[index];" << std::endl;
    ss << " if (d < 3 * dim) {" << std::endl;
    ss << "x_diff[index] = x_acts_diff[index] * x_act * (Dtype(1) - x_act);"
       << std::endl;
    ss << "} else {" << std::endl;
    ss << "x_diff[index] = x_acts_diff[index] * (Dtype(1) - x_act * x_act);"
       << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
     KernelArgs args;
     args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                       "nthreads", KERNEL_ARG_CONST));
     args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                       "dim", KERNEL_ARG_CONST));
     args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                       "c_prev", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
     args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                       "x", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
     args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                       "c", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
     args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                       "h", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
     args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                       "cont", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
     args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                       "c_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
     args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                       "h_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
     args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                       "c_prev_diff", KERNEL_ARG_GLOBAL_MEM));
     args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                       "x_diff", KERNEL_ARG_GLOBAL_MEM));
     ss << this->device_program_->function("LSTMUnitBackward", args);
     ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
     ss << "const int_tp n = index / dim;" << std::endl;
     ss << "const int_tp d = index % dim;" << std::endl;
     ss << this->device_program_->global_ptr("const Dtype", "x_offset")
        << " = x + 4 * dim * n;" << std::endl;
     ss << "const Dtype i = x_offset[d];" << std::endl;
     ss << "const Dtype f = x_offset[1 * dim + d];" << std::endl;
     ss << "const Dtype o = x_offset[2 * dim + d];" << std::endl;
     ss << "const Dtype g = x_offset[3 * dim + d];" << std::endl;
     ss << "const Dtype c_prev_t = c_prev[index];" << std::endl;
     ss << "const Dtype c_t = c[index];" << std::endl;
     ss << "const Dtype tanh_c = " << tanh<Dtype>("c_t") << ";" << std::endl;
     ss << this->device_program_->global_ptr("Dtype", "c_prev_diff_t")
        << " = c_prev_diff + index;" << std::endl;
     ss << this->device_program_->global_ptr("Dtype", "x_diff_offset")
        << " = x_diff + 4 * dim * n;" << std::endl;
     ss << this->device_program_->global_ptr("Dtype", "i_diff")
        << " = x_diff_offset + d;" << std::endl;
     ss << this->device_program_->global_ptr("Dtype", "f_diff")
        << " = x_diff_offset + 1 * dim + d;" << std::endl;
     ss << this->device_program_->global_ptr("Dtype", "o_diff")
        << " = x_diff_offset + 2 * dim + d;" << std::endl;
     ss << this->device_program_->global_ptr("Dtype", "g_diff")
        << " = x_diff_offset + 3 * dim + d;" << std::endl;
     ss << "const Dtype c_term_diff = "
        << "c_diff[index] + h_diff[index] * o * (1 - tanh_c * tanh_c);"
        << std::endl;
     ss << "const Dtype cont_n = cont[n];" << std::endl;
     ss << "*c_prev_diff_t = cont_n * c_term_diff * f;" << std::endl;
     ss << "*i_diff = c_term_diff * g;" << std::endl;
     ss << "*f_diff = cont_n * c_term_diff * c_prev_t;" << std::endl;
     ss << "*o_diff = h_diff[index] * tanh_c;" << std::endl;
     ss << "*g_diff = c_term_diff * i;" << std::endl;
     ss << "}" << std::endl;
     ss << "}" << std::endl;
   }

  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}

template<typename Dtype, typename MItype, typename MOtype>
void LSTMUnitLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                         const vector<Blob<MItype>*>& bottom,
                                         const vector<Blob<MOtype>*>& top) {
  const uint_tp count = top[1]->count();
  vptr<const Dtype> c_prev = bottom[0]->gpu_data();
  vptr<const Dtype> x = bottom[1]->gpu_data();
  vptr<const Dtype> cont = bottom[2]->gpu_data();
  vptr<const Dtype> x_acts = X_acts_.mutable_gpu_data();
  vptr<const Dtype> c = top[0]->mutable_gpu_data();
  vptr<const Dtype> h = top[1]->mutable_gpu_data();
  const int_tp x_count = bottom[1]->count();

  {
    shared_ptr<DeviceKernel> kernel =
                            this->device_program_->GetKernel("LSTMActsForward");
    kernel->add_arg(&x_count);
    kernel->add_arg(&hidden_dim_);
    kernel->add_arg(&x);
    kernel->add_arg(&x_acts);

    vector<size_t> work_size(1, x_count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }

  {
    shared_ptr<DeviceKernel> kernel =
                            this->device_program_->GetKernel("LSTMUnitForward");
    kernel->add_arg(&count);
    kernel->add_arg(&hidden_dim_);
    kernel->add_arg(&c_prev);
    kernel->add_arg(&x_acts);
    kernel->add_arg(&cont);
    kernel->add_arg(&c);
    kernel->add_arg(&h);

    vector<size_t> work_size(1, count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void LSTMUnitLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                         const vector<Blob<MOtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<MItype>*>& bottom) {
  CHECK(!propagate_down[2]) << "Cannot backpropagate to sequence indicators.";
  if (!propagate_down[0] && !propagate_down[1]) { return; }

  const uint_tp count = top[1]->count();
  vptr<const Dtype> c_prev = bottom[0]->gpu_data();
  vptr<const Dtype> x_acts = X_acts_.gpu_data();
  vptr<const Dtype> cont = bottom[2]->gpu_data();
  vptr<const Dtype> c = top[0]->gpu_data();
  vptr<const Dtype> h = top[1]->gpu_data();
  vptr<const Dtype> c_diff = top[0]->gpu_diff();
  vptr<const Dtype> h_diff = top[1]->gpu_diff();
  vptr<const Dtype> c_prev_diff = bottom[0]->mutable_gpu_diff();
  vptr<const Dtype> x_acts_diff = X_acts_.mutable_gpu_diff();
  const int_tp x_count = bottom[1]->count();
  vptr<const Dtype> x_diff = bottom[1]->mutable_gpu_diff();

  {
    shared_ptr<DeviceKernel> kernel =
                           this->device_program_->GetKernel("LSTMUnitBackward");
    kernel->add_arg(&count);
    kernel->add_arg(&hidden_dim_);
    kernel->add_arg(&c_prev);
    kernel->add_arg(&x_acts);
    kernel->add_arg(&c);
    kernel->add_arg(&h);
    kernel->add_arg(&cont);
    kernel->add_arg(&c_diff);
    kernel->add_arg(&h_diff);
    kernel->add_arg(&c_prev_diff);
    kernel->add_arg(&x_acts_diff);

    vector<size_t> work_size(1, count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }

  {
    shared_ptr<DeviceKernel> kernel =
                           this->device_program_->GetKernel("LSTMActsBackward");
    kernel->add_arg(&x_count);
    kernel->add_arg(&hidden_dim_);
    kernel->add_arg(&x_acts);
    kernel->add_arg(&x_acts_diff);
    kernel->add_arg(&x_diff);

    vector<size_t> work_size(1, x_count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(LSTMUnitLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LSTMUnitLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LSTMUnitLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(LSTMUnitLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LSTMUnitLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LSTMUnitLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(LSTMUnitLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LSTMUnitLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(LSTMUnitLayer, Backward_gpu,
                                  (double), (double), (double));
}  // namespace caffe
