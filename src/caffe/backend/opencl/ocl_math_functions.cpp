#include "caffe/backend/opencl/ocl_math.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"

namespace caffe {

#ifdef USE_OPENCL

void ocl_device::add_half(const uint_tp N, vptr<half_float::half> a,
                      vptr<half_float::half> b, vptr<half_float::half> y) {
}

void ocl_device::add_float(const uint_tp N, vptr<float> a,
                       vptr<float> b, vptr<float> y) {
}

void ocl_device::add_double(const uint_tp N, vptr<double> a,
                       vptr<double> b, vptr<double> y) {
}

void ocl_device::sub_half(const uint_tp N, vptr<half_float::half> a,
                      vptr<half_float::half> b, vptr<half_float::half> y) {
}

void ocl_device::sub_float(const uint_tp N, vptr<float> a, vptr<float> b,
                       vptr<float> y) {
}

void ocl_device::sub_double(const uint_tp N, vptr<double> a, vptr<double> b,
                        vptr<double> y) {
}

void ocl_device::mul_half(const uint_tp N, vptr<half_float::half> a,
                      vptr<half_float::half> b, vptr<half_float::half> y) {
}

void ocl_device::mul_float(const uint_tp N, vptr<float> a,
                      vptr<float> b, vptr<float> y) {
}

void ocl_device::mul_double(const uint_tp N, vptr<double> a,
                      vptr<double> b, vptr<double> y) {
}

void ocl_device::abs_half(const uint_tp n, vptr<half_float::half> a,
                  vptr<half_float::half> y) {
}

 void ocl_device::abs_float(const uint_tp n, vptr<float> a, vptr<float> y) {
}

 void ocl_device::abs_double(const uint_tp n, vptr<double> a, vptr<double> y) {
}

 void ocl_device::exp_half(const uint_tp n, vptr<half_float::half> a,
                       vptr<half_float::half> y) {
}

 void ocl_device::exp_float(const uint_tp n, vptr<float> a, vptr<float> y) {
}

 void ocl_device::exp_double(const uint_tp n, vptr<double> a, vptr<double> y) {
}

 void ocl_device::log_half(const uint_tp n, vptr<half_float::half> a,
                       vptr<half_float::half> y) {
}

 void ocl_device::log_float(const uint_tp n, vptr<float> a, vptr<float> y) {
}

 void ocl_device::log_double(const uint_tp n, vptr<double> a, vptr<double> y) {
}

 void ocl_device::powx_half(const uint_tp n, vptr<half_float::half> a,
                        const half_float::half b,
                        vptr<half_float::half> y) {
}

 void ocl_device::powx_float(const uint_tp n, vptr<float> a, const float b,
                         vptr<float> y) {
}

 void ocl_device::powx_double(const uint_tp n, vptr<double> a, const double b,
                          vptr<double> y) {
}

 void ocl_device::sqrt_half(const uint_tp n, vptr<half_float::half> a,
                        vptr<half_float::half> y) {
}

 void ocl_device::sqrt_float(const uint_tp n, vptr<float> a, vptr<float> y) {
}

 void ocl_device::sqrt_double(const uint_tp n, vptr<double> a, vptr<double> y) {
}

void ocl_device::memset(const int_tp ctx_id, const uint_tp N, const int_tp alpha,
                     vptr<float> X, const int_tp offX) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  // OpenCL Version >= 1.2 approach
  // clEnqueueFillBuffer(ctx.get_queue().handle().get(),
  //  X, &alpha, sizeof(int_tp),
  //                     offX, N, 0, NULL, NULL);
  // OpenCL Version < 1.2 fallback
  typedef float float;
  viennacl::ocl::kernel &oclk_fill = program.get_kernel(
      CL_KERNEL_SELECT("fillbuffer"));
  viennacl::ocl::enqueue(
      oclk_fill(static_cast<int_tp>(N), static_cast<unsigned char>(alpha),
                WrapHandle(X, &ctx), offX),
      ctx.get_queue());
}

// Copy from OpenCL buffer to main memory
void ocl_device::memcpy(const uint_tp N, const vptr<float> X, const int_tp offX,
                         void *Y, viennacl::ocl::context *ctx) {
  if (Y != NULL) {
    clEnqueueReadBuffer(ctx->get_queue().handle().get(), X, CL_TRUE, offX, N, Y,
                        0,
                        NULL,
                        NULL);
  }
}

// Copy from main memory to OpenCL buffer
void ocl_device::memcpy(const uint_tp N, const void* X, vptr<float> Y,
                         const int_tp offY, viennacl::ocl::context *ctx) {
  if (X != NULL) {
    clEnqueueWriteBuffer(ctx->get_queue().handle().get(), Y,
    CL_TRUE,
                         offY, N, X, 0, NULL, NULL);
  }
}

// Copy from OpenCL to OpenCL buffer
void ocl_device::memcpy(const uint_tp N, const vptr<float> X, const int_tp offX,
                         vptr<float> Y, const int_tp offY,
                         viennacl::ocl::context *ctx) {
  clEnqueueCopyBuffer(ctx->get_queue().handle().get(), X, Y, offX, offY, N, 0,
  NULL,
                      NULL);
}

template<typename float>
void greentea_copy(const int_tp N, const vptr<float> X, const int_tp offX, float* Y,
                   viennacl::ocl::context *ctx) {
  ocl_device::memcpy(sizeof(float) * N, X, offX * sizeof(float), Y, ctx);
}

template<typename float>
void greentea_copy(const int_tp N, const float* X, vptr<float> Y, const int_tp offY,
                   viennacl::ocl::context *ctx) {
  ocl_device::memcpy(sizeof(float) * N, X, Y, offY * sizeof(float), ctx);
}

// Copy from OpenCL buffer to OpenCL buffer
template<typename float>
void greentea_copy(const int_tp N, const vptr<float> X, const int_tp offX, vptr<float> Y,
                   const int_tp offY, viennacl::ocl::context *ctx) {
  ocl_device::memcpy(sizeof(float) * N, X, offX * sizeof(float), Y,
                      offY * sizeof(float), ctx);
}


void ocl_device::mul(const int_tp ctx_id, const int_tp N, const vptr<float> a,
                      const int_tp offa, const vptr<float> b, const int_tp offb,
                      vptr<float> y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_mul = program.get_kernel(CL_KERNEL_SELECT("mul"));
  viennacl::ocl::enqueue(
      oclk_mul(N, WrapHandle(a, &ctx), offa, WrapHandle(b, &ctx), offb,
               WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

#ifdef USE_GPU_HALF
template void ocl_device::mul<half_float::half>(const int_tp ctx_id, const int_tp N,
                                     const vptr<float> a, const int_tp offa,
                                     const vptr<float> b, const int_tp offb,
                                     vptr<float> y, const int_tp offy);
#endif
template void ocl_device::mul<float>(const int_tp ctx_id, const int_tp N,
                                      const vptr<float> a, const int_tp offa,
                                      const vptr<float> b, const int_tp offb,
                                      vptr<float> y, const int_tp offy);
template void ocl_device::mul<double>(const int_tp ctx_id, const int_tp N,
                                       const vptr<float> a, const int_tp offa,
                                       const vptr<float> b, const int_tp offb,
                                       vptr<float> y, const int_tp offy);

template<typename float>
void ocl_device::div(const int_tp ctx_id, const int_tp N, const vptr<float> a,
                      const int_tp offa, const vptr<float> b, const int_tp offb,
                      vptr<float> y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_div = program.get_kernel(CL_KERNEL_SELECT("div"));
  viennacl::ocl::enqueue(
      oclk_div(N, WrapHandle(a, &ctx), offa, WrapHandle(b, &ctx), offb,
               WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}






template<typename float>
void ocl_device::set(const int_tp ctx_id, const int_tp N, const float alpha,
                      vptr<float> Y, const int_tp offY) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();
  // OpenCL Version >= 1.2 approach
  // clEnqueueFillBuffer(ctx.get_queue().handle().get(),
  //                  Y, &alpha, sizeof(float),
  //                  offY, N, 0, NULL, NULL);

  // OpenCL Version < 1.2 fallback
  viennacl::ocl::kernel &oclk_fill = program.get_kernel(
      CL_KERNEL_SELECT("fill"));
  viennacl::ocl::enqueue(oclk_fill(N, fixup_arg_type(alpha),
                         WrapHandle(Y, &ctx), offY),
                         ctx.get_queue());
}

template<typename float>
void ocl_device::add_scalar(const int_tp ctx_id, const int_tp N,
                             const float alpha, vptr<float> Y, const int_tp offY) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_add_scalar = program.get_kernel(
      CL_KERNEL_SELECT("add_scalar"));
  viennacl::ocl::enqueue(oclk_add_scalar(N, fixup_arg_type(alpha),
                         WrapHandle(Y, &ctx), offY),
                         ctx.get_queue());
}

template<typename float>
void ocl_device::add(const int_tp ctx_id, const int_tp n, const vptr<float> a,
                      const int_tp offa, const vptr<float> b, const int_tp offb,
                      vptr<float> y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_add = program.get_kernel(CL_KERNEL_SELECT("add"));
  viennacl::ocl::enqueue(
      oclk_add(n, WrapHandle(a, &ctx), offa, WrapHandle(b, &ctx), offb,
               WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template<typename float>
void ocl_device::sub(const int_tp ctx_id, const int_tp n, const vptr<float> a,
                      const int_tp offa, const vptr<float> b, const int_tp offb,
                      vptr<float> y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_sub = program.get_kernel(CL_KERNEL_SELECT("sub"));
  viennacl::ocl::enqueue(
      oclk_sub(n, WrapHandle(a, &ctx), offa, WrapHandle(b, &ctx), offb,
               WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template<typename float>
void ocl_device::abs(const int_tp ctx_id, const int_tp N, const vptr<float> a,
                      const int_tp offa, vptr<float> y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_abs = program.get_kernel(CL_KERNEL_SELECT("abs"));
  viennacl::ocl::enqueue(
      oclk_abs(N, WrapHandle(a, &ctx), offa, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template<typename float>
void ocl_device::exp(const int_tp ctx_id, const int_tp N, const vptr<float> a,
                      const int_tp offa, vptr<float> y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_exp = program.get_kernel(CL_KERNEL_SELECT("exp"));
  viennacl::ocl::enqueue(
      oclk_exp(N, WrapHandle(a, &ctx), offa, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template<typename float>
void ocl_device::sqrt(const int_tp ctx_id, const int_tp n,
                       const vptr<float> a, const int_tp offa,
                       vptr<float> y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_sqrt = program.get_kernel(
      CL_KERNEL_SELECT("sqrt"));
  viennacl::ocl::enqueue(
      oclk_sqrt(n, WrapHandle(a, &ctx), offa, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}


template<typename float>
void ocl_device::powx(const int_tp ctx_id, const int_tp N, const vptr<float> a,
                       const int_tp offa, const float alpha, vptr<float> y,
                       const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_powx = program.get_kernel(
      CL_KERNEL_SELECT("powx"));
  viennacl::ocl::enqueue(
      oclk_powx(N, WrapHandle(a, &ctx), offa, fixup_arg_type(alpha),
                WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}


template<typename float>
void ocl_device::log(const int_tp ctx_id, const int_tp N, const vptr<float> a,
                      const int_tp offa, vptr<float> y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_log = program.get_kernel(CL_KERNEL_SELECT("log"));
  viennacl::ocl::enqueue(
      oclk_log(N, WrapHandle(a, &ctx), offa, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template<typename float>
void ocl_device::sign(const int_tp ctx_id, const int_tp n, const vptr<float> x,
int_tp offx,
                       vptr<float> y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_sign = program.get_kernel(
      CL_KERNEL_SELECT("sign"));
  viennacl::ocl::enqueue(
      oclk_sign(n, WrapHandle(x, &ctx), offx, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}


template<typename float>
void ocl_device::sgnbit(const int_tp ctx_id, const int_tp n, const vptr<float> x,
int_tp offx,
                         vptr<float> y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_sgnbit = program.get_kernel(
      CL_KERNEL_SELECT("sgnbit"));
  viennacl::ocl::enqueue(
      oclk_sgnbit(n, WrapHandle(x, &ctx), offx, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

void ocl_device::rng_uniform(const int_tp ctx_id, const int_tp n, vptr<float> r,
int_tp offr) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  std::vector<uint_tp> random(n);  //NOLINT
  caffe_rng_uniform(n, &random[0]);
  ocl_device::memcpy(sizeof(uint_tp) * n, &random[0], r, offr, &ctx);
}

template<typename float>
void ocl_device::rng_uniform(const int_tp ctx_id, const int_tp n,
                              const float a, const float b, vptr<float> r,
                              const int_tp offr) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  std::vector<float> random(n);  // NOLINT
  caffe_rng_uniform(n, a, b, &random[0]);
  ocl_device::memcpy(sizeof(float) * n, &random[0], r, offr, &ctx);
}


template<typename float>
void ocl_device::rng_gaussian(const int_tp ctx_id, const int_tp n,
                               const float mu, const float sigma, vptr<float> r,
                               const int_tp offr) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  std::vector<float> random(n);  // NOLINT
  caffe_rng_gaussian(n, mu, sigma, &random[0]);
  ocl_device::memcpy(sizeof(float) * n, &random[0], r, offr, &ctx);
}


void ocl_device::add_scalar_half(const uint_tp N, const half_float::half alpha,
                        vptr<half_float::half> X) {
}

void ocl_device::add_scalar_float(const uint_tp N, const float alpha,
                        vptr<float> X) {
}

void ocl_device::add_scalar_double(const uint_tp N, const double alpha,
                        vptr<double> X) {
}

void ocl_device::div_half(const uint_tp N, vptr<half_float::half> a,
                      vptr<half_float::half> b, vptr<half_float::half> y) {
}

void ocl_device::div_float(const uint_tp N, vptr<float> a, vptr<float> b,
                       vptr<float> y) {
}

void ocl_device::div_double(const uint_tp N, vptr<double> a, vptr<double> b,
                        vptr<double> y) {
}



}  // namespace caffe
#endif  // USE_OPENCL
