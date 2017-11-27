#include "caffe/backend/device_program.hpp"
#include "caffe/backend/device.hpp"
#include "caffe/util/type_utils.hpp"

namespace caffe {

void Device::memset(const uint_tp n, const char alpha, vptr<char> x) {
  shared_ptr<DeviceKernel> kernel
             = math_program_->GetKernel("gpu_memset");

  kernel->add_arg(&n);
  kernel->add_arg(&alpha);
  kernel->add_arg(&x);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<typename Dtype>
void Device::set(const uint_tp n, const Dtype alpha, vptr<Dtype> x) {
  shared_ptr<DeviceKernel> kernel
             = math_program_->GetKernel("gpu_set_" + safe_type_name<Dtype>());

  kernel->add_arg(&n);
  kernel->add_arg(&alpha);
  kernel->add_arg(x);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<>
void Device::set(const uint_tp n, const half_fp alpha,
                 vptr<half_fp> x);
template<>
void Device::set(const uint_tp n, const float alpha, vptr<float> x);
template<>
void Device::set(const uint_tp n, const double alpha, vptr<double> x);


template<typename Dtype>
void Device::add_scalar(const uint_tp n, const Dtype alpha, vptr<Dtype> x) {
  shared_ptr<DeviceKernel> kernel
      = math_program_->GetKernel("gpu_add_scalar_" + safe_type_name<Dtype>());

  kernel->add_arg(&n);
  kernel->add_arg(&alpha);
  kernel->add_arg(x);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<>
void Device::add_scalar(const uint_tp n, const half_fp alpha,
                vptr<half_fp> x);
template<>
void Device::add_scalar(const uint_tp n, const float alpha, vptr<float> x);
template<>
void Device::add_scalar(const uint_tp n, const double alpha, vptr<double> x);


template<typename Dtype>
void Device::scal(const uint_tp n, const Dtype alpha, vptr<Dtype> x) {
  shared_ptr<DeviceKernel> kernel
      = math_program_->GetKernel("gpu_scal_" + safe_type_name<Dtype>());

  kernel->add_arg(&n);
  kernel->add_arg(&alpha);
  kernel->add_arg(x);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<>
void Device::scal(const uint_tp n, const half_fp alpha,
                  vptr<half_fp> x);
template<>
void Device::scal(const uint_tp n, const float alpha, vptr<float> x);
template<>
void Device::scal(const uint_tp n, const double alpha, vptr<double> x);


template<typename Dtype>
void Device::add(const uint_tp n, vptr<const Dtype> a,
                 vptr<const Dtype> b, vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_program_->GetKernel("gpu_add_" + safe_type_name<Dtype>());

  kernel->add_arg(&n);
  kernel->add_arg(a);
  kernel->add_arg(b);
  kernel->add_arg(y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<>
void Device::add(const uint_tp n, vptr<const half_fp> a,
                 vptr<const half_fp> b, vptr<half_fp> y);
template<>
void Device::add(const uint_tp n, vptr<const float> a, vptr<const float> b,
                 vptr<float> y);
template<>
void Device::add(const uint_tp n, vptr<const double> a, vptr<const double> b,
                 vptr<double> y);


template<typename Dtype>
void Device::sub(const uint_tp n, vptr<const Dtype> a, vptr<const Dtype> b,
                 vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_program_->GetKernel("gpu_sub_" + safe_type_name<Dtype>());

  kernel->add_arg(&n);
  kernel->add_arg(a);
  kernel->add_arg(b);
  kernel->add_arg(y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<>
void Device::sub(const uint_tp n, vptr<const half_fp> a,
                 vptr<const half_fp> b, vptr<half_fp> y);
template<>
void Device::sub(const uint_tp n, vptr<const float> a, vptr<const float> b,
                 vptr<float> y);
template<>
void Device::sub(const uint_tp n, vptr<const double> a, vptr<const double> b,
                 vptr<double> y);


template<typename Dtype>
void Device::mul(const uint_tp n, vptr<const Dtype> a, vptr<const Dtype> b,
                 vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_program_->GetKernel("gpu_mul_" + safe_type_name<Dtype>());

  kernel->add_arg(&n);
  kernel->add_arg(a);
  kernel->add_arg(b);
  kernel->add_arg(y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<>
void Device::mul(const uint_tp n, vptr<const half_fp> a,
                 vptr<const half_fp> b, vptr<half_fp> y);
template<>
void Device::mul(const uint_tp n, vptr<const float> a, vptr<const float> b,
                 vptr<float> y);
template<>
void Device::mul(const uint_tp n, vptr<const double> a, vptr<const double> b,
                 vptr<double> y);


template<typename Dtype>
void Device::div(const uint_tp n, vptr<const Dtype> a, vptr<const Dtype> b,
                 vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_program_->GetKernel("gpu_div_" + safe_type_name<Dtype>());

  kernel->add_arg(&n);
  kernel->add_arg(a);
  kernel->add_arg(b);
  kernel->add_arg(y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<>
void Device::div(const uint_tp n, vptr<const half_fp> a,
                 vptr<const half_fp> b, vptr<half_fp> y);
template<>
void Device::div(const uint_tp n, vptr<const float> a, vptr<const float> b,
                 vptr<float> y);
template<>
void Device::div(const uint_tp n, vptr<const double> a, vptr<const double> b,
                 vptr<double> y);


template<typename Dtype>
void Device::abs(const uint_tp n, vptr<const Dtype> a, vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_program_->GetKernel("gpu_abs_" + safe_type_name<Dtype>());

  kernel->add_arg(&n);
  kernel->add_arg(a);
  kernel->add_arg(y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<>
void Device::abs(const uint_tp n, vptr<const half_fp> a,
                 vptr<half_fp> y);
template<>
void Device::abs(const uint_tp n, vptr<const float> a, vptr<float> y);
template<>
void Device::abs(const uint_tp n, vptr<const double> a, vptr<double> y);


template<typename Dtype>
void Device::exp(const uint_tp n, vptr<const Dtype> a, vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_program_->GetKernel("gpu_exp_" + safe_type_name<Dtype>());

  kernel->add_arg(&n);
  kernel->add_arg(a);
  kernel->add_arg(y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<>
void Device::exp(const uint_tp n, vptr<const half_fp> a,
                 vptr<half_fp> y);
template<>
void Device::exp(const uint_tp n, vptr<const float> a, vptr<float> y);
template<>
void Device::exp(const uint_tp n, vptr<const double> a, vptr<double> y);


template<typename Dtype>
void Device::log(const uint_tp n, vptr<const Dtype> a, vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_program_->GetKernel("gpu_log_" + safe_type_name<Dtype>());

  kernel->add_arg(&n);
  kernel->add_arg(a);
  kernel->add_arg(y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<>
void Device::log(const uint_tp n, vptr<const half_fp> a,
                 vptr<half_fp> y);
template<>
void Device::log(const uint_tp n, vptr<const float> a, vptr<float> y);
template<>
void Device::log(const uint_tp n, vptr<const double> a, vptr<double> y);


template<typename Dtype>
void Device::powx(const uint_tp n, vptr<const Dtype> a, const Dtype b,
                  vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_program_->GetKernel("gpu_powx_" + safe_type_name<Dtype>());

  kernel->add_arg(&n);
  kernel->add_arg(a);
  kernel->add_arg(b);
  kernel->add_arg(y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<>
void Device::powx(const uint_tp n, vptr<const half_fp> a,
                  const half_fp b, vptr<half_fp> y);
template<>
void Device::powx(const uint_tp n, vptr<const float> a, const float b,
                  vptr<float> y);
template<>
void Device::powx(const uint_tp n, vptr<const double> a, const double b,
                  vptr<double> y);


template<typename Dtype>
void Device::sqrt(const uint_tp n, vptr<const Dtype> a, vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_program_->GetKernel("gpu_sqrt_" + safe_type_name<Dtype>());

  kernel->add_arg(&n);
  kernel->add_arg(a);
  kernel->add_arg(y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<>
void Device::sqrt(const uint_tp n, vptr<const half_fp> a,
                  vptr<half_fp> y);
template<>
void Device::sqrt(const uint_tp n, vptr<const float> a, vptr<float> y);
template<>
void Device::sqrt(const uint_tp n, vptr<const double> a, vptr<double> y);


template<typename Dtype>
void Device::sign(const uint_tp n, vptr<const Dtype> x, vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_program_->GetKernel("gpu_sign_" + safe_type_name<Dtype>());

  kernel->add_arg(&n);
  kernel->add_arg(x);
  kernel->add_arg(y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<>
void Device::sign(const uint_tp n, vptr<const half_fp> x,
                  vptr<half_fp> y);
template<>
void Device::sign(const uint_tp n, vptr<const float> x, vptr<float> y);
template<>
void Device::sign(const uint_tp n, vptr<const double> x, vptr<double> y);


template<typename Dtype>
void Device::sgnbit(const uint_tp n, vptr<const Dtype> x, vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_program_->GetKernel("gpu_signbit_" + safe_type_name<Dtype>());

  kernel->add_arg(&n);
  kernel->add_arg(&x);
  kernel->add_arg(&y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<>
void Device::sgnbit(const uint_tp n, vptr<const half_fp> x,
                    vptr<half_fp> y);
template<>
void Device::sgnbit(const uint_tp n, vptr<const float> x, vptr<float> y);
template<>
void Device::sgnbit(const uint_tp n, vptr<const double> x, vptr<double> y);


template<typename Dtype>
string create_source(Device* dev,
                           shared_ptr<DeviceProgram> program) {
  stringstream ss;

  ss << program->define_type<Dtype>("Dtype");

  // Set
  {
    KernelArgs args;
    args.push_back(program->create_kernel_arg<uint_tp>("n",
                      KERNEL_ARG_CONST));
    args.push_back(program->create_kernel_arg<Dtype>("alpha",
                      KERNEL_ARG_CONST));
    args.push_back(program->create_kernel_arg<Dtype>("y",
                      KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_MEM_OFFSET));
    ss << program->function("gpu_set_" + safe_type_name<Dtype>(),
                                          args);
    ss << program->kernel_loop("uint_tp", "index", "n");
    ss << "y[index] = alpha;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  // Add Scalar
  {
    KernelArgs args;
    args.push_back(program->create_kernel_arg<uint_tp>("n",
                      KERNEL_ARG_CONST));
    args.push_back(program->create_kernel_arg<Dtype>("alpha",
                      KERNEL_ARG_CONST));
    args.push_back(program->create_kernel_arg<Dtype>("y",
                      KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_MEM_OFFSET));
    ss << program->function("gpu_add_scalar_"
                                               + safe_type_name<Dtype>(), args);
    ss << program->kernel_loop("uint_tp", "index", "n");
    ss << "y[index] += alpha;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  // Add, Sub, Mul, Div
  vector<string> op_names = {"add", "sub", "mul", "div"};
  vector<string> ops = {"+", "-", "*", "/"};
  for (uint_tp i = 0; i < ops.size(); ++i) {
    string op_name = op_names[i];
    string op = ops[i];
    KernelArgs args;
    args.push_back(program->create_kernel_arg<uint_tp>("n",
                      KERNEL_ARG_CONST));
    args.push_back(program->create_kernel_arg<Dtype>("a",
                      KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM
                      | KERNEL_ARG_MEM_OFFSET));
    args.push_back(program->create_kernel_arg<Dtype>("b",
                      KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM
                      | KERNEL_ARG_MEM_OFFSET));
    args.push_back(program->create_kernel_arg<Dtype>("y",
                      KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_MEM_OFFSET));
    ss << program->function("gpu_" + op_name + "_"
                                          + safe_type_name<Dtype>(), args);
    ss << program->kernel_loop("uint_tp", "index", "n");
    ss << "y[index] = a[index] " << op << " b[index];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  // Powx
  {
    KernelArgs args;
    args.push_back(program->create_kernel_arg<uint_tp>("n",
                      KERNEL_ARG_CONST));
    args.push_back(program->create_kernel_arg<Dtype>("a",
                      KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM
                      | KERNEL_ARG_MEM_OFFSET));
    args.push_back(program->create_kernel_arg<Dtype>("alpha",
                      KERNEL_ARG_CONST));
    args.push_back(program->create_kernel_arg<Dtype>("y",
                      KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_MEM_OFFSET));
    ss << program->function("gpu_powx_"
                                          + safe_type_name<Dtype>(), args);
    ss << program->kernel_loop("uint_tp", "index", "n");
    ss << "if(alpha == 2.0) {" << std::endl;
    string abs_fun;
    if (dev->backend() == BACKEND_CUDA) {
      abs_fun = "abs";
    } else {
      abs_fun = "fabs";
    }
    ss << "y[index] = pow((Dtype)" << abs_fun
       << "(a[index]), (Dtype)alpha);" << std::endl;
    ss << "} else {" << std::endl;
    ss << "y[index] = pow((Dtype)a[index], (Dtype)alpha);" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  // Abs, Exp, Log
  vector<string> func_names = {"exp", "log", "sqrt", "abs"};
  vector<string> funcs = {"exp", "log", "sqrt"};
  if (dev->backend() == BACKEND_CUDA) {
    funcs.push_back("abs");
  } else {
    funcs.push_back("fabs");
  }
  for (uint_tp i = 0; i < funcs.size(); ++i) {
    string func_name = func_names[i];
    string func = funcs[i];
    KernelArgs args;
    args.push_back(program->create_kernel_arg<uint_tp>("n",
                      KERNEL_ARG_CONST));
    args.push_back(program->create_kernel_arg<Dtype>("a",
                      KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM
                      | KERNEL_ARG_MEM_OFFSET));
    args.push_back(program->create_kernel_arg<Dtype>("y",
                      KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_MEM_OFFSET));
    ss << program->function("gpu_" + func_name + "_"
                                          + safe_type_name<Dtype>(), args);
    ss << program->kernel_loop("uint_tp", "index", "n");
    ss << "y[index] = " << func << "(a[index]);" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  // Sign
  {
    KernelArgs args;
    args.push_back(program->create_kernel_arg<uint_tp>("n",
                      KERNEL_ARG_CONST));
    args.push_back(program->create_kernel_arg<Dtype>("x",
                      KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM
                      | KERNEL_ARG_MEM_OFFSET));
    args.push_back(program->create_kernel_arg<Dtype>("y",
                      KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_MEM_OFFSET));
    ss << program->function("gpu_sign_" + safe_type_name<Dtype>(),
                                          args);
    ss << program->kernel_loop("uint_tp", "index", "n");
    ss << "y[index] = (0.0 < x[index]) - (x[index] < 0.0);" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  // Signbit
  {
    KernelArgs args;
    args.push_back(program->create_kernel_arg<uint_tp>("n",
                      KERNEL_ARG_CONST));
    args.push_back(program->create_kernel_arg<Dtype>("x",
                      KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM
                      | KERNEL_ARG_MEM_OFFSET));
    args.push_back(program->create_kernel_arg<Dtype>("y",
                      KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_MEM_OFFSET));
    ss << program->function("gpu_signbit_"
                                          + safe_type_name<Dtype>(), args);
    ss << program->kernel_loop("uint_tp", "index", "n");
    ss << "y[index] = signbit(x[index]);" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  return ss.str();
}

template<>
string create_source<half_fp>(Device* dev,
    shared_ptr<DeviceProgram> program);
template<>
string create_source<float>(Device* dev,
    shared_ptr<DeviceProgram> program);
template<>
string create_source<double>(Device* dev,
    shared_ptr<DeviceProgram> program);


void Device::CreateMathProgram() {
  this->math_program_ = this->CreateProgram();
  stringstream ss;

  ss << this->math_program_->setup();

  // Memset
  {
    KernelArgs args;
    args.push_back(this->math_program_->create_kernel_arg<uint_tp>("n",
                      KERNEL_ARG_CONST));
    args.push_back(this->math_program_->create_kernel_arg<char>("alpha",
                      KERNEL_ARG_CONST));
    args.push_back(this->math_program_->create_kernel_arg<char>("y",
                      KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_MEM_OFFSET));
    ss << this->math_program_->function("gpu_memset", args);
    ss << this->math_program_->kernel_loop("uint_tp", "index", "n");
    ss << "y[index] = alpha;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

#ifdef USE_HALF
  ss << create_source<half_fp>(this, this->math_program_);
#endif

#ifdef USE_SINGLE
  ss << create_source<float>(this, this->math_program_);
#endif

#ifdef USE_DOUBLE
  ss << create_source<double>(this, this->math_program_);
#endif

  this->math_program_->set_source(ss.str());
  this->math_program_->Compile(true, true);
}



}  // namespace caffe
