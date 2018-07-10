#include "caffe/backend/device_program.hpp"
#include "caffe/backend/device.hpp"
#include "caffe/util/type_utils.hpp"

namespace caffe {

void Device::null_kernel(float arg) {
  shared_ptr<DeviceKernel> kernel
        = math_programs_[AUX_DATA_INDEX]->GetKernel("caffe_gpu_null_kernel");

  kernel->add_arg(&arg);

  vector<size_t> work_size(1, 1);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

void Device::memset(const uint_tp n, const char alpha, vptr<char> x) {
  shared_ptr<DeviceKernel> kernel
        = math_programs_[AUX_DATA_INDEX]->GetKernel("caffe_gpu_memset");

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
   = math_programs_[data_type_index<Dtype>()]->GetKernel("caffe_gpu_set");

  kernel->add_arg(&n);
  kernel->add_arg(&alpha);
  kernel->add_arg(&x);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template void Device::set(const uint_tp n, const half_fp alpha,
                          vptr<half_fp> x);
template void Device::set(const uint_tp n, const float alpha, vptr<float> x);
template void Device::set(const uint_tp n, const double alpha, vptr<double> x);
template void Device::set(const uint_tp n, const int8_t alpha, vptr<int8_t> x);
template void Device::set(const uint_tp n, const int16_t alpha,
                          vptr<int16_t> x);
template void Device::set(const uint_tp n, const int32_t alpha,
                          vptr<int32_t> x);
template void Device::set(const uint_tp n, const int64_t alpha,
                          vptr<int64_t> x);
template void Device::set(const uint_tp n, const uint8_t alpha,
                          vptr<uint8_t> x);
template void Device::set(const uint_tp n, const uint16_t alpha,
                          vptr<uint16_t> x);
template void Device::set(const uint_tp n, const uint32_t alpha,
                          vptr<uint32_t> x);
template void Device::set(const uint_tp n, const uint64_t alpha,
                          vptr<uint64_t> x);

template<typename Dtype>
void Device::add_scalar(const uint_tp n, const Dtype alpha, vptr<Dtype> x) {
  shared_ptr<DeviceKernel> kernel
      = math_programs_[data_type_index<Dtype>()]
                       ->GetKernel("caffe_gpu_add_scalar");

  kernel->add_arg(&n);
  kernel->add_arg(&alpha);
  kernel->add_arg(&x);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template
void Device::add_scalar(const uint_tp n, const half_fp alpha, vptr<half_fp> x);
template
void Device::add_scalar(const uint_tp n, const float alpha, vptr<float> x);
template
void Device::add_scalar(const uint_tp n, const double alpha, vptr<double> x);
template
void Device::add_scalar(const uint_tp n, const uint8_t alpha,
                        vptr<uint8_t> x);
template
void Device::add_scalar(const uint_tp n, const uint16_t alpha,
                        vptr<uint16_t> x);
template
void Device::add_scalar(const uint_tp n, const uint32_t alpha,
                        vptr<uint32_t> x);
template
void Device::add_scalar(const uint_tp n, const uint64_t alpha,
                        vptr<uint64_t> x);

template<typename Dtype>
void Device::scal(const uint_tp n, const Dtype alpha, vptr<Dtype> x) {
  shared_ptr<DeviceKernel> kernel
      = math_programs_[data_type_index<Dtype>()]
                       ->GetKernel("caffe_gpu_scal");

  kernel->add_arg(&n);
  kernel->add_arg(&alpha);
  kernel->add_arg(&x);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template
void Device::scal(const uint_tp n, const half_fp alpha, vptr<half_fp> x);
template
void Device::scal(const uint_tp n, const float alpha, vptr<float> x);
template
void Device::scal(const uint_tp n, const double alpha, vptr<double> x);
template
void Device::scal(const uint_tp n, const uint8_t alpha, vptr<uint8_t> x);
template
void Device::scal(const uint_tp n, const uint16_t alpha, vptr<uint16_t> x);
template
void Device::scal(const uint_tp n, const uint32_t alpha, vptr<uint32_t> x);
template
void Device::scal(const uint_tp n, const uint64_t alpha, vptr<uint64_t> x);

template<typename Dtype>
void Device::add(const uint_tp n, vptr<const Dtype> a,
                 vptr<const Dtype> b, vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_programs_[data_type_index<Dtype>()]
                       ->GetKernel("caffe_gpu_add");

  kernel->add_arg(&n);
  kernel->add_arg(&a);
  kernel->add_arg(&b);
  kernel->add_arg(&y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template
void Device::add(const uint_tp n, vptr<const half_fp> a,
                 vptr<const half_fp> b, vptr<half_fp> y);
template
void Device::add(const uint_tp n, vptr<const float> a, vptr<const float> b,
                 vptr<float> y);
template
void Device::add(const uint_tp n, vptr<const double> a, vptr<const double> b,
                 vptr<double> y);
template
void Device::add(const uint_tp n, vptr<const uint8_t> a,
                 vptr<const uint8_t> b, vptr<uint8_t> y);
template
void Device::add(const uint_tp n, vptr<const uint16_t> a,
                 vptr<const uint16_t> b, vptr<uint16_t> y);
template
void Device::add(const uint_tp n, vptr<const uint32_t> a,
                 vptr<const uint32_t> b, vptr<uint32_t> y);
template
void Device::add(const uint_tp n, vptr<const uint64_t> a,
                 vptr<const uint64_t> b, vptr<uint64_t> y);


template<typename Dtype>
void Device::sub(const uint_tp n, vptr<const Dtype> a, vptr<const Dtype> b,
                 vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_programs_[data_type_index<Dtype>()]
                       ->GetKernel("caffe_gpu_sub");

  kernel->add_arg(&n);
  kernel->add_arg(&a);
  kernel->add_arg(&b);
  kernel->add_arg(&y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template
void Device::sub(const uint_tp n, vptr<const half_fp> a,
                 vptr<const half_fp> b, vptr<half_fp> y);
template
void Device::sub(const uint_tp n, vptr<const float> a, vptr<const float> b,
                 vptr<float> y);
template
void Device::sub(const uint_tp n, vptr<const double> a, vptr<const double> b,
                 vptr<double> y);
template
void Device::sub(const uint_tp n, vptr<const uint8_t> a, vptr<const uint8_t> b,
                 vptr<uint8_t> y);
template
void Device::sub(const uint_tp n, vptr<const uint16_t> a,
                 vptr<const uint16_t> b, vptr<uint16_t> y);
template
void Device::sub(const uint_tp n, vptr<const uint32_t> a,
                 vptr<const uint32_t> b, vptr<uint32_t> y);
template
void Device::sub(const uint_tp n, vptr<const uint64_t> a,
                 vptr<const uint64_t> b, vptr<uint64_t> y);


template<typename Dtype>
void Device::mul(const uint_tp n, vptr<const Dtype> a, vptr<const Dtype> b,
                 vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_programs_[data_type_index<Dtype>()]
                       ->GetKernel("caffe_gpu_mul");

  kernel->add_arg(&n);
  kernel->add_arg(&a);
  kernel->add_arg(&b);
  kernel->add_arg(&y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template
void Device::mul(const uint_tp n, vptr<const half_fp> a,
                 vptr<const half_fp> b, vptr<half_fp> y);
template
void Device::mul(const uint_tp n, vptr<const float> a, vptr<const float> b,
                 vptr<float> y);
template
void Device::mul(const uint_tp n, vptr<const double> a, vptr<const double> b,
                 vptr<double> y);
template
void Device::mul(const uint_tp n, vptr<const uint8_t> a,
                 vptr<const uint8_t> b, vptr<uint8_t> y);
template
void Device::mul(const uint_tp n, vptr<const uint16_t> a,
                 vptr<const uint16_t> b, vptr<uint16_t> y);
template
void Device::mul(const uint_tp n, vptr<const uint32_t> a,
                 vptr<const uint32_t> b, vptr<uint32_t> y);
template
void Device::mul(const uint_tp n, vptr<const uint64_t> a,
                 vptr<const uint64_t> b, vptr<uint64_t> y);


template<typename Dtype>
void Device::div(const uint_tp n, vptr<const Dtype> a, vptr<const Dtype> b,
                 vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_programs_[data_type_index<Dtype>()]
                       ->GetKernel("caffe_gpu_div");

  kernel->add_arg(&n);
  kernel->add_arg(&a);
  kernel->add_arg(&b);
  kernel->add_arg(&y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template
void Device::div(const uint_tp n, vptr<const half_fp> a,
                 vptr<const half_fp> b, vptr<half_fp> y);
template
void Device::div(const uint_tp n, vptr<const float> a, vptr<const float> b,
                 vptr<float> y);
template
void Device::div(const uint_tp n, vptr<const double> a, vptr<const double> b,
                 vptr<double> y);
template
void Device::div(const uint_tp n, vptr<const uint8_t> a, vptr<const uint8_t> b,
                 vptr<uint8_t> y);
template
void Device::div(const uint_tp n, vptr<const uint16_t> a, vptr<const uint16_t> b,
                 vptr<uint16_t> y);
template
void Device::div(const uint_tp n, vptr<const uint32_t> a,
                 vptr<const uint32_t> b, vptr<uint32_t> y);
template
void Device::div(const uint_tp n, vptr<const uint64_t> a,
                 vptr<const uint64_t> b, vptr<uint64_t> y);


template<typename Dtype>
void Device::abs(const uint_tp n, vptr<const Dtype> a, vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_programs_[data_type_index<Dtype>()]
                       ->GetKernel("caffe_gpu_abs");

  kernel->add_arg(&n);
  kernel->add_arg(&a);
  kernel->add_arg(&y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template
void Device::abs(const uint_tp n, vptr<const half_fp> a,
                 vptr<half_fp> y);
template
void Device::abs(const uint_tp n, vptr<const float> a, vptr<float> y);
template
void Device::abs(const uint_tp n, vptr<const double> a, vptr<double> y);
template
void Device::abs(const uint_tp n, vptr<const uint8_t> a,
                 vptr<uint8_t> y);
template
void Device::abs(const uint_tp n, vptr<const uint16_t> a,
                 vptr<uint16_t> y);
template
void Device::abs(const uint_tp n, vptr<const uint32_t> a,
                 vptr<uint32_t> y);
template
void Device::abs(const uint_tp n, vptr<const uint64_t> a,
                 vptr<uint64_t> y);


template<typename Dtype>
void Device::exp(const uint_tp n, vptr<const Dtype> a, vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_programs_[data_type_index<Dtype>()]
                       ->GetKernel("caffe_gpu_exp");

  kernel->add_arg(&n);
  kernel->add_arg(&a);
  kernel->add_arg(&y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template
void Device::exp(const uint_tp n, vptr<const half_fp> a, vptr<half_fp> y);
template
void Device::exp(const uint_tp n, vptr<const float> a, vptr<float> y);
template
void Device::exp(const uint_tp n, vptr<const double> a, vptr<double> y);
template
void Device::exp(const uint_tp n, vptr<const uint8_t> a, vptr<uint8_t> y);
template
void Device::exp(const uint_tp n, vptr<const uint16_t> a, vptr<uint16_t> y);
template
void Device::exp(const uint_tp n, vptr<const uint32_t> a, vptr<uint32_t> y);
template
void Device::exp(const uint_tp n, vptr<const uint64_t> a, vptr<uint64_t> y);


template<typename Dtype>
void Device::log(const uint_tp n, vptr<const Dtype> a, vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_programs_[data_type_index<Dtype>()]
                       ->GetKernel("caffe_gpu_log");

  kernel->add_arg(&n);
  kernel->add_arg(&a);
  kernel->add_arg(&y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template
void Device::log(const uint_tp n, vptr<const half_fp> a,
                 vptr<half_fp> y);
template
void Device::log(const uint_tp n, vptr<const float> a, vptr<float> y);
template
void Device::log(const uint_tp n, vptr<const double> a, vptr<double> y);
template
void Device::log(const uint_tp n, vptr<const uint8_t> a, vptr<uint8_t> y);
template
void Device::log(const uint_tp n, vptr<const uint16_t> a, vptr<uint16_t> y);
template
void Device::log(const uint_tp n, vptr<const uint32_t> a, vptr<uint32_t> y);
template
void Device::log(const uint_tp n, vptr<const uint64_t> a, vptr<uint64_t> y);


template<typename Dtype>
void Device::powx(const uint_tp n, vptr<const Dtype> a, const Dtype b,
                  vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_programs_[data_type_index<Dtype>()]->GetKernel("caffe_gpu_powx");

  kernel->add_arg(&n);
  kernel->add_arg(&a);
  kernel->add_arg(&b);
  kernel->add_arg(&y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template
void Device::powx(const uint_tp n, vptr<const half_fp> a,
                  const half_fp b, vptr<half_fp> y);
template
void Device::powx(const uint_tp n, vptr<const float> a, const float b,
                  vptr<float> y);
template
void Device::powx(const uint_tp n, vptr<const double> a, const double b,
                  vptr<double> y);
template
void Device::powx(const uint_tp n, vptr<const uint8_t> a,
                  const uint8_t b, vptr<uint8_t> y);
template
void Device::powx(const uint_tp n, vptr<const uint16_t> a,
                  const uint16_t b, vptr<uint16_t> y);
template
void Device::powx(const uint_tp n, vptr<const uint32_t> a,
                  const uint32_t b, vptr<uint32_t> y);
template
void Device::powx(const uint_tp n, vptr<const uint64_t> a,
                  const uint64_t b, vptr<uint64_t> y);


template<typename Dtype>
void Device::sqrt(const uint_tp n, vptr<const Dtype> a, vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_programs_[data_type_index<Dtype>()]
                       ->GetKernel("caffe_gpu_sqrt");

  kernel->add_arg(&n);
  kernel->add_arg(&a);
  kernel->add_arg(&y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template
void Device::sqrt(const uint_tp n, vptr<const half_fp> a, vptr<half_fp> y);
template
void Device::sqrt(const uint_tp n, vptr<const float> a, vptr<float> y);
template
void Device::sqrt(const uint_tp n, vptr<const double> a, vptr<double> y);
template
void Device::sqrt(const uint_tp n, vptr<const uint8_t> a, vptr<uint8_t> y);
template
void Device::sqrt(const uint_tp n, vptr<const uint16_t> a, vptr<uint16_t> y);
template
void Device::sqrt(const uint_tp n, vptr<const uint32_t> a, vptr<uint32_t> y);
template
void Device::sqrt(const uint_tp n, vptr<const uint64_t> a, vptr<uint64_t> y);

template<typename Dtype>
void Device::sign(const uint_tp n, vptr<const Dtype> x, vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_programs_[data_type_index<Dtype>()]
                       ->GetKernel("caffe_gpu_sign");

  kernel->add_arg(&n);
  kernel->add_arg(&x);
  kernel->add_arg(&y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template
void Device::sign(const uint_tp n, vptr<const half_fp> x, vptr<half_fp> y);
template
void Device::sign(const uint_tp n, vptr<const float> x, vptr<float> y);
template
void Device::sign(const uint_tp n, vptr<const double> x, vptr<double> y);
template
void Device::sign(const uint_tp n, vptr<const uint8_t> x, vptr<uint8_t> y);
template
void Device::sign(const uint_tp n, vptr<const uint16_t> x, vptr<uint16_t> y);
template
void Device::sign(const uint_tp n, vptr<const uint32_t> x, vptr<uint32_t> y);
template
void Device::sign(const uint_tp n, vptr<const uint64_t> x, vptr<uint64_t> y);

template<typename Dtype>
void Device::sgnbit(const uint_tp n, vptr<const Dtype> x, vptr<Dtype> y) {
  shared_ptr<DeviceKernel> kernel
      = math_programs_[data_type_index<Dtype>()]
                       ->GetKernel("caffe_gpu_signbit");

  kernel->add_arg(&n);
  kernel->add_arg(&x);
  kernel->add_arg(&y);

  vector<size_t> work_size(1, n);
  vector<size_t> group;
  vector<size_t> local;
  this->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template
void Device::sgnbit(const uint_tp n, vptr<const half_fp> x, vptr<half_fp> y);
template
void Device::sgnbit(const uint_tp n, vptr<const float> x, vptr<float> y);
template
void Device::sgnbit(const uint_tp n, vptr<const double> x, vptr<double> y);
template
void Device::sgnbit(const uint_tp n, vptr<const uint8_t> x, vptr<uint8_t> y);
template
void Device::sgnbit(const uint_tp n, vptr<const uint16_t> x, vptr<uint16_t> y);
template
void Device::sgnbit(const uint_tp n, vptr<const uint32_t> x, vptr<uint32_t> y);
template
void Device::sgnbit(const uint_tp n, vptr<const uint64_t> x, vptr<uint64_t> y);


template<typename Dtype, typename MItype, typename MOtype>
string create_set_source(Device* dev, shared_ptr<DeviceProgram> program) {
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
    ss << program->function("caffe_gpu_set", args);
    ss << program->kernel_loop("uint_tp", "index", "n");
    ss << "y[index] = alpha;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }
  return ss.str();
}


template<typename Dtype, typename MItype, typename MOtype>
string create_source(Device* dev, shared_ptr<DeviceProgram> program) {
  stringstream ss;

  ss << program->helper_functions<Dtype>();
  ss << program->define_type<Dtype>("Dtype");

  // Add Scalar
  {
    KernelArgs args;
    args.push_back(program->create_kernel_arg<uint_tp>("n",
                      KERNEL_ARG_CONST));
    args.push_back(program->create_kernel_arg<Dtype>("alpha",
                      KERNEL_ARG_CONST));
    args.push_back(program->create_kernel_arg<Dtype>("y",
                      KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_MEM_OFFSET));
    ss << program->function("caffe_gpu_add_scalar", args);
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
    ss << program->function("caffe_gpu_" + op_name, args);
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
    ss << program->function("caffe_gpu_powx", args);
    ss << program->kernel_loop("uint_tp", "index", "n");
    ss << "if (alpha == (Dtype)2) {" << std::endl;
    string abs_fun;
    if (dev->backend() == BACKEND_CUDA) {
      abs_fun = "abs";
    } else {
      abs_fun = "fabs";
    }
    if (is_unsigned_integer_type<Dtype>()) {
      ss << "y[index] = (Dtype)(pow((float)abs(a[index]), (float)alpha));"
         << std::endl;
    } else {
      ss << "y[index] = pow((Dtype)" << abs_fun << "(a[index]), (Dtype)alpha);"
      << std::endl;
    }
    ss << "} else {" << std::endl;
    if (is_unsigned_integer_type<Dtype>()) {
      ss << "y[index] = (Dtype)(pow((float)a[index], (float)alpha));"
         << std::endl;
    } else {
      ss << "y[index] = pow((Dtype)a[index], (Dtype)alpha);" << std::endl;
    }
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  // Abs, Exp, Log
  vector<string> func_names = {"exp", "log", "sqrt", "abs"};
  vector<string> funcs = {"exp", "log", "sqrt"};
  if (dev->backend() == BACKEND_CUDA
      || is_unsigned_integer_type<Dtype>()) {
    // CUDA and integer absolute value
    funcs.push_back("abs");
  } else {
    // OpenCL float absolute value
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
    ss << program->function("caffe_gpu_" + func_name, args);
    ss << program->kernel_loop("uint_tp", "index", "n");
    if (is_unsigned_integer_type<Dtype>() && !(func_name == "abs")) {
      ss << "y[index] = (Dtype)" << func << "((float)a[index]);" << std::endl;
    } else {
      ss << "y[index] = " << func << "(a[index]);" << std::endl;
    }
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
    ss << program->function("caffe_gpu_sign", args);
    ss << program->kernel_loop("uint_tp", "index", "n");
    ss << "y[index] = ((Dtype)0 < x[index]) - (x[index] < (Dtype)0);"
       << std::endl;
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
    ss << program->function("caffe_gpu_signbit", args);
    ss << program->kernel_loop("uint_tp", "index", "n");
    if (is_unsigned_integer_type<Dtype>()) {
      ss << "y[index] = (Dtype)(x[index] < (Dtype)0);" << std::endl;
    } else {
      ss << "y[index] = signbit(x[index]);" << std::endl;
    }
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  return ss.str();
}

void Device::CreateMathProgram() {
  for (int_tp i = 0; i < PROTO_DATA_INDEX_MAX; ++i) {
    this->math_programs_.push_back(this->CreateProgram());
    stringstream ss;
    ss << this->math_programs_[i]->setup();

    switch (i) {
      case AUX_DATA_INDEX: {
        // Memset
        {
          KernelArgs args;
          args.push_back(this->math_programs_[i]
                         ->create_kernel_arg<uint_tp>("n",
                         KERNEL_ARG_CONST));
          args.push_back(this->math_programs_[i]
                         ->create_kernel_arg<char>("alpha",
                         KERNEL_ARG_CONST));
          args.push_back(this->math_programs_[i]
                         ->create_kernel_arg<char>("y",
                         KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_MEM_OFFSET));
          ss << this->math_programs_[i]->function("caffe_gpu_memset", args);
          ss << this->math_programs_[i]->kernel_loop("uint_tp", "index", "n");
          ss << "y[index] = alpha;" << std::endl;
          ss << "}" << std::endl;
          ss << "}" << std::endl;
        }

        // Null kernel
        {
          KernelArgs args;
          args.push_back(this->math_programs_[i]
                          ->create_kernel_arg<float>("arg", KERNEL_ARG_NONE));
          ss << this->math_programs_[i]
                                     ->function("caffe_gpu_null_kernel", args);
          ss << "float out = arg;" << std::endl;
          ss << "}" << std::endl;
        }
        break;
      }
      case HALF_DATA_INDEX: {
#ifdef USE_HALF
        ss << "#ifdef HALF_SUPPORT_AVAILABLE" << std::endl;
        ss << create_set_source<half_fp, half_fp, half_fp>(this,
                                                       this->math_programs_[i]);
        ss << create_source<half_fp, half_fp, half_fp>(this,
                                                       this->math_programs_[i]);
        ss << "#endif  // HALF_SUPPORT_AVAILABLE" << std::endl;
#endif  // USE_HALF
        break;
      }
      case FLOAT_DATA_INDEX: {
#ifdef USE_SINGLE
        ss << create_set_source<float, float, float>(this,
                                                       this->math_programs_[i]);
        ss << create_source<float, float, float>(this,
                                                       this->math_programs_[i]);
#endif  // USE_SINGLE
        break;
      }
      case DOUBLE_DATA_INDEX: {
#ifdef USE_DOUBLE
        ss << "#ifdef DOUBLE_SUPPORT_AVAILABLE" << std::endl;
        ss << create_set_source<double, double, double>(this,
                                                       this->math_programs_[i]);
        ss << create_source<double, double, double>(this,
                                                       this->math_programs_[i]);
        ss << "#endif  // DOUBLE_SUPPORT_AVAILABLE" << std::endl;
#endif  // USE_DOUBLE
        break;
      }
      case UINT8_DATA_INDEX: {
        ss << create_set_source<uint8_t, uint8_t, uint8_t>(this,
                                                       this->math_programs_[i]);
        ss << create_source<uint8_t, uint8_t, uint8_t>(this,
                                                       this->math_programs_[i]);
        break;
      }
      case UINT16_DATA_INDEX: {
        ss << create_set_source<uint16_t, uint16_t, uint16_t>(this,
                                                       this->math_programs_[i]);
        ss << create_source<uint16_t, uint16_t, uint16_t>(this,
                                                       this->math_programs_[i]);
        break;
      }
      case UINT32_DATA_INDEX: {
        ss << create_set_source<uint32_t, uint32_t, uint32_t>(this,
                                                       this->math_programs_[i]);
        ss << create_source<uint32_t, uint32_t, uint32_t>(this,
                                                       this->math_programs_[i]);
        break;
      }
      case UINT64_DATA_INDEX: {
        ss << create_set_source<uint64_t, uint64_t, uint64_t>(this,
                                                       this->math_programs_[i]);
        ss << create_source<uint64_t, uint64_t, uint64_t>(this,
                                                       this->math_programs_[i]);
        break;
      }
      case INT8_DATA_INDEX: {
        ss << create_set_source<int8_t, int8_t, int8_t>(this,
                                                       this->math_programs_[i]);
        break;
      }
      case INT16_DATA_INDEX: {
        ss << create_set_source<int16_t, int16_t, int16_t>(this,
                                                       this->math_programs_[i]);
        break;
      }
      case INT32_DATA_INDEX: {
        ss << create_set_source<int32_t, int32_t, int32_t>(this,
                                                       this->math_programs_[i]);
        break;
      }
      case INT64_DATA_INDEX: {
        ss << create_set_source<int64_t, int64_t, int64_t>(this,
                                                       this->math_programs_[i]);
        break;
      }
    }
    this->math_programs_[i]->set_source(ss.str());
    this->math_programs_[i]->Compile(true, true);
  }
}

}  // namespace caffe
