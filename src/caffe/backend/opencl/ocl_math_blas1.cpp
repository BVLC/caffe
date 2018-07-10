#ifdef USE_OPENCL

#include "caffe/backend/vptr.hpp"
#include "caffe/backend/opencl/ocl_math.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"
#include "caffe/backend/opencl/ocl_dev_ptr.hpp"

namespace caffe {

#ifdef USE_HALF
void OclDevice::axpy_half(const uint_tp n, const half_fp alpha,
                          vptr<const half_fp> x,
                          vptr<half_fp> y,
                          const QuantizerValues* const alpha_quant,
                          const QuantizerValues* const x_quant,
                          const QuantizerValues* const y_quant) {
  uint_tp offX = x.get_ocl_off();
  uint_tp offY = y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

#if defined(USE_CLBLAS)

  cl_command_queue queue = ctx.get_queue().handle().get();

  OPENCL_CL_BLAS_CHECK(
      clblasHaxpy(n, alpha, x.get_ocl_mem(), offX,
          1, y.get_ocl_mem(), offY, 1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

  cl_command_queue queue = ctx.get_queue().handle().get();

  const size_t incX = 1;
  const size_t incY = 1;

  OPENCL_CLBLAST_CHECK(
    clblast::Axpy<clblast::half>(
      n,
      clblast::FloatToHalf(alpha),
      x.get_ocl_mem(), offX, incX,
      y.get_ocl_mem(), offY, incY,
      &queue));

#else  // default (ViennaCL)
  Device::axpy_half(n, alpha, x, y);
#endif  // clBLAS, CLBlast, or default (ViennaCL)
}
void OclDevice::axpby_half(const uint_tp n, const half_fp alpha,
                   vptr<const half_fp> x,
                   const half_fp beta, vptr<half_fp> y,
                   const QuantizerValues* const alpha_quant,
                   const QuantizerValues* const x_quant,
                   const QuantizerValues* const beta_quant,
                   const QuantizerValues* const y_quant) {
  this->scal_half(n, beta, y);
  this->axpy_half(n, alpha, x, y, alpha_quant, x_quant, y_quant);
}
void OclDevice::scal_half(const uint_tp n, const half_fp alpha,
                  vptr<half_fp> x) {
  uint_tp offX = x.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

#if defined(USE_CLBLAS)

  cl_command_queue queue = ctx.get_queue().handle().get();

  OPENCL_CL_BLAS_CHECK(clblasHscal(n, alpha, x.get_ocl_mem(), offX,
          1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

  cl_command_queue queue = ctx.get_queue().handle().get();

  const size_t incx = 1;

  OPENCL_CLBLAST_CHECK(
    clblast::Scal<clblast::half>(
      n,
      clblast::FloatToHalf(alpha),
      x.get_ocl_mem(), offX, incx,
      &queue));

#else  // default (ViennaCL)
  Device::scal_half(n, alpha, x);
#endif  // clBLAS, CLBlast, or default (ViennaCL)
}
void OclDevice::dot_half(const uint_tp n, vptr<const half_fp> x,
                         vptr<const half_fp> y,
                         half_fp* out) {
  uint_tp offX = x.get_ocl_off();
  uint_tp offY = y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

#if defined(USE_CLBLAS)

  cl_command_queue queue = ctx.get_queue().handle().get();

  cl_int err;
  cl_mem gpuout = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      sizeof(half_fp), NULL, &err);
  cl_mem scratch = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      n * sizeof(half_fp), NULL, &err);

  OPENCL_CL_BLAS_CHECK(
      clblasHdot(n, gpuout, 0, x.get_ocl_mem(), offX, 1, y.get_ocl_mem(),
          offY, 1, scratch, 1, &queue, 0, NULL, NULL));


  shared_ptr<ocl_dev_ptr<half_fp> > oclptr_gpuout
                    = make_shared<ocl_dev_ptr<half_fp> >(gpuout);
  vptr<half_fp> vptr_gpuout(oclptr_gpuout);
  this->memcpy(sizeof(half_fp), vptr<void>(vptr_gpuout), out);

  clReleaseMemObject(gpuout);
  clReleaseMemObject(scratch);

#elif defined(USE_CLBLAST)

  cl_command_queue queue = ctx.get_queue().handle().get();

  cl_int err = CL_SUCCESS;
  cl_mem Z = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
                            sizeof(half_fp), NULL, &err);

  const size_t offZ = 0;
  const size_t incX = 1;
  const size_t incY = 1;

  OPENCL_CLBLAST_CHECK(
    clblast::Dot<clblast::half>(
      n,
      Z, offZ,
      x.get_ocl_mem(), offX, incX,
      y.get_ocl_mem(), offY, incY,
      &queue));

  shared_ptr<ocl_dev_ptr<half_fp> > oclptrZ
                    = make_shared<ocl_dev_ptr<half_fp> >(Z, offZ);
  vptr<half_fp> vptrZ(oclptrZ);
  this->memcpy(sizeof(half_fp), vptr<void>(vptrZ), out);
  clReleaseMemObject(Z);

#else  // default (ViennaCL)
  Device::dot_half(n, x, y, out);
#endif  // clBLAS, CLBlast, or default (ViennaCL)
}
void OclDevice::asum_half(const uint_tp n, vptr<const half_fp> x,
                           half_fp* y) {
  uint_tp offX = x.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

#if defined(USE_CLBLAS)

  cl_command_queue queue = ctx.get_queue().handle().get();

  cl_int err;
  cl_mem gpuout = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      sizeof(half_fp), NULL, &err);
  cl_mem scratch = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      n * sizeof(half_fp), NULL, &err);

  OPENCL_CL_BLAS_CHECK(
      clblasHasum(n, gpuout, 0, x.get_ocl_mem(), offX, 1,
          scratch, 1, &queue, 0, NULL, NULL));

  shared_ptr<ocl_dev_ptr<half_fp> > oclptr_gpuout
                     = make_shared<ocl_dev_ptr<half_fp> >(gpuout);
  vptr<half_fp> vptr_gpuout(oclptr_gpuout);
  this->memcpy(sizeof(half_fp), vptr<void>(vptr_gpuout), y);

  clReleaseMemObject(gpuout);
  clReleaseMemObject(scratch);

#elif defined(USE_CLBLAST)

  cl_command_queue queue = ctx.get_queue().handle().get();

  cl_int err = CL_SUCCESS;
  cl_mem Z = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
    sizeof(half_fp), NULL, &err);

  const size_t offZ = 0;
  const size_t incX = 1;

  OPENCL_CLBLAST_CHECK(
    clblast::Asum<clblast::half>(
      n,
      Z, offZ,
      x.get_ocl_mem(), offX, incX,
      &queue));

  shared_ptr<ocl_dev_ptr<half_fp> > oclptrZ
                    = make_shared<ocl_dev_ptr<half_fp> >(Z, offZ);
  vptr<half_fp> vptrZ(oclptrZ);
  this->memcpy(sizeof(half_fp), vptr<void>(vptrZ), y);

  clReleaseMemObject(Z);

#else  // default (ViennaCL)
  Device::asum_half(n, x, y);
#endif  // clBLAS, CLBlast, or default (ViennaCL)
}
void OclDevice::scale_half(const uint_tp n, const half_fp alpha,
                           vptr<const half_fp> x,
                           vptr<half_fp> y) {
  uint_tp offX = x.get_ocl_off();
  uint_tp offY = y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    half_fp* Xptr = reinterpret_cast<half_fp*>(
        clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(half_fp) * offX, sizeof(half_fp) * n, 0,
        NULL, NULL, NULL));
    half_fp* Yptr = reinterpret_cast<half_fp*>(
        clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), y.get_ocl_mem(), true, CL_MAP_WRITE,
        sizeof(half_fp) * offY, sizeof(half_fp) * n, 0,
        NULL, NULL, NULL));

    caffe_scale<half_fp>(n, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                            Xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), y.get_ocl_mem(),
                            Yptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(
        clblasHcopy(n, x.get_ocl_mem(), offX, 1, y.get_ocl_mem(), offY, 1, 1,
                    &queue, 0, NULL, NULL));
    OPENCL_CL_BLAS_CHECK(
        clblasHscal(n, alpha, y.get_ocl_mem(), offY, 1, 1, &queue,
                    0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Copy<clblast::half>(
        n,
        x.get_ocl_mem(), offX, incX,
        y.get_ocl_mem(), offY, incY,
        &queue));
    OPENCL_CLBLAST_CHECK(
      clblast::Scal<clblast::half>(
        n,
        clblast::FloatToHalf(alpha),
        y.get_ocl_mem(), offY, incY,
        &queue));

#else  // default (ViennaCL)
    Device::scale_half(n, alpha, x, y);
#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}
#endif  // USE_HALF


#ifdef USE_SINGLE
void OclDevice::axpy_float(const uint_tp n, const float alpha,
                           vptr<const float> x, vptr<float> y,
                           const QuantizerValues* const alpha_quant,
                           const QuantizerValues* const x_quant,
                           const QuantizerValues* const y_quant) {

  uint_tp offX = x.get_ocl_off();
  uint_tp offY = y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offX, sizeof(float) * n, 0, NULL, NULL, NULL));
    float* Yptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), y.get_ocl_mem(), true, CL_MAP_WRITE,
        sizeof(float) * offY, sizeof(float) * n, 0, NULL, NULL, NULL));

    caffe_axpy<float>(n, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                            Xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), y.get_ocl_mem(),
                            Yptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(
        clblasSaxpy(n, alpha, x.get_ocl_mem(), offX,
            1, y.get_ocl_mem(), offY, 1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Axpy<float>(
        n,
        alpha,
        x.get_ocl_mem(), offX, incX,
        y.get_ocl_mem(), offY, incY,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(x.get_ocl_mem(),
                     size_type(n), size_type(offX), difference_type(1), ctx);
    viennacl::vector_base<float, size_t, ptrdiff_t> v2(y.get_ocl_mem(),
                     size_type(n), size_type(offY), difference_type(1), ctx);
    v2 += alpha * v1;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}
void OclDevice::axpby_float(const uint_tp n, const float alpha,
                   vptr<const float> x, const float beta, vptr<float> y,
                   const QuantizerValues* const alpha_quant,
                   const QuantizerValues* const x_quant,
                   const QuantizerValues* const beta_quant,
                   const QuantizerValues* const y_quant) {
  this->scal_float(n, beta, y);
  this->axpy_float(n, alpha, x, y, alpha_quant, x_quant, y_quant);
}
void OclDevice::scal_float(const uint_tp n, const float alpha, vptr<float> x) {
  uint_tp offX = x.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x.get_ocl_mem(), true,
        CL_MAP_READ | CL_MAP_WRITE, sizeof(float) * offX, sizeof(float) * n, 0,
        NULL, NULL, NULL));

    caffe_scal<float>(n, alpha, xptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                            xptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(clblasSscal(n, alpha, x.get_ocl_mem(), offX,
            1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incx = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Scal<float>(
        n,
        alpha,
        x.get_ocl_mem(), offX, incx,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(x.get_ocl_mem(),
                      size_type(n), size_type(offX), difference_type(1), ctx);
    v1 *= alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}
void OclDevice::dot_float(const uint_tp n, vptr<const float> x,
                          vptr<const float> y, float* out) {
  uint_tp offX = x.get_ocl_off();
  uint_tp offY = y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offX, sizeof(float) * n, 0, NULL, NULL, NULL));
    float* Yptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), y.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offY, sizeof(float) * n, 0, NULL, NULL, NULL));

    *out = caffe_dot<float>(n, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                            Xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), y.get_ocl_mem(),
                            Yptr, 0, NULL, NULL);

  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err;
    cl_mem gpuout = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
        sizeof(float), NULL, &err);
    cl_mem scratch = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
        n * sizeof(float), NULL, &err);

    OPENCL_CL_BLAS_CHECK(
        clblasSdot(n, gpuout, 0, x.get_ocl_mem(), offX, 1, y.get_ocl_mem(),
            offY, 1, scratch, 1, &queue, 0, NULL, NULL));

    shared_ptr<ocl_dev_ptr<float> > oclptr_gpuout
                                = make_shared<ocl_dev_ptr<float> >(gpuout);
    vptr<float> vptr_gpuout(oclptr_gpuout);
    this->memcpy(sizeof(float), vptr<void>(vptr_gpuout), out);
    clReleaseMemObject(gpuout);
    clReleaseMemObject(scratch);

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err = CL_SUCCESS;
    cl_mem Z = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      sizeof(float), NULL, &err);

    const size_t offZ = 0;
    const size_t incX = 1;
    const size_t incY = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Dot<float>(
        n,
        Z, offZ,
        x.get_ocl_mem(), offX, incX,
        y.get_ocl_mem(), offY, incY,
        &queue));

    shared_ptr<ocl_dev_ptr<float> > oclptrZ
                               = make_shared<ocl_dev_ptr<float> >(Z, offZ);
    vptr<float> vptrZ(oclptrZ);
    this->memcpy(sizeof(float), vptr<void>(vptrZ), out);
    clReleaseMemObject(Z);

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(x.get_ocl_mem(),
                   size_type(n), size_type(offX), difference_type(1), ctx);
    viennacl::vector_base<float, size_t, ptrdiff_t> v2(y.get_ocl_mem(),
                   size_type(n), size_type(offY), difference_type(1), ctx);

    *out = viennacl::linalg::inner_prod(v1, v2);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}
void OclDevice::asum_float(const uint_tp n, vptr<const float> x, float* y) {
  uint_tp offX = x.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offX, sizeof(float) * n, 0, NULL, NULL, NULL));

    *y = caffe_asum<float>(n, Xptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                            Xptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err;
    cl_mem gpuout = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
        sizeof(float), NULL, &err);
    cl_mem scratch = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
        n * sizeof(float), NULL, &err);

    OPENCL_CL_BLAS_CHECK(
        clblasSasum(n, gpuout, 0, x.get_ocl_mem(), offX, 1,
            scratch, 1, &queue, 0, NULL, NULL));

    shared_ptr<ocl_dev_ptr<float> > oclptr_gpuout
                                = make_shared<ocl_dev_ptr<float> >(gpuout);
    vptr<float> vptr_gpuout(oclptr_gpuout);
    this->memcpy(sizeof(float), vptr<void>(vptr_gpuout), y);

    clReleaseMemObject(gpuout);
    clReleaseMemObject(scratch);

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err = CL_SUCCESS;
    cl_mem Z = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      sizeof(float), NULL, &err);

    const size_t offZ = 0;
    const size_t incX = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Asum<float>(
        n,
        Z, offZ,
        x.get_ocl_mem(), offX, incX,
        &queue));

    shared_ptr<ocl_dev_ptr<float> > oclptrZ
                    = make_shared<ocl_dev_ptr<float> >(Z, offZ);
    vptr<float> vptrZ(oclptrZ);
    this->memcpy(sizeof(float), vptr<void>(vptrZ), y);
    clReleaseMemObject(Z);

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(x.get_ocl_mem(),
                        size_type(n), size_type(offX), difference_type(1), ctx);

    *y = viennacl::linalg::norm_1(v1);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}
void OclDevice::scale_float(const uint_tp n, const float alpha,
                             vptr<const float> x, vptr<float> y) {
  uint_tp offX = x.get_ocl_off();
  uint_tp offY = y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offX, sizeof(float) * n, 0, NULL, NULL, NULL));
    float* Yptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), y.get_ocl_mem(), true, CL_MAP_WRITE,
        sizeof(float) * offY, sizeof(float) * n, 0, NULL, NULL, NULL));

    caffe_scale<float>(n, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                            Xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), y.get_ocl_mem(),
                            Yptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(
        clblasScopy(n, x.get_ocl_mem(), offX, 1, y.get_ocl_mem(), offY, 1, 1,
                    &queue, 0, NULL, NULL));
    OPENCL_CL_BLAS_CHECK(
        clblasSscal(n, alpha, y.get_ocl_mem(), offY, 1, 1, &queue,
                    0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Copy<float>(
        n,
        x.get_ocl_mem(), offX, incX,
        y.get_ocl_mem(), offY, incY,
        &queue));
    OPENCL_CLBLAST_CHECK(
      clblast::Scal<float>(
        n,
        alpha,
        y.get_ocl_mem(), offY, incY,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(x.get_ocl_mem(),
                        size_type(n), size_type(offX), difference_type(1), ctx);
    viennacl::vector_base<float, size_t, ptrdiff_t> v2(y.get_ocl_mem(),
                        size_type(n), size_type(offY), difference_type(1), ctx);

    v2 = v1 * alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}
#endif  // USE_SINGLE


#ifdef USE_DOUBLE
void OclDevice::axpy_double(const uint_tp n, const double alpha,
                            vptr<const double> x, vptr<double> y) {
  uint_tp offX = x.get_ocl_off();
  uint_tp offY = y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* Xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offX, sizeof(double) * n, 0, NULL, NULL, NULL));
    double* Yptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), y.get_ocl_mem(), true, CL_MAP_WRITE,
        sizeof(double) * offY, sizeof(double) * n, 0, NULL, NULL, NULL));

    caffe_axpy<double>(n, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                            Xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), y.get_ocl_mem(),
                            Yptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(
        clblasDaxpy(n, alpha, x.get_ocl_mem(), offX,
            1, y.get_ocl_mem(), offY, 1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Axpy<double>(
        n,
        alpha,
        x.get_ocl_mem(), offX, incX,
        y.get_ocl_mem(), offY, incY,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(x.get_ocl_mem(),
                     size_type(n), size_type(offX), difference_type(1), ctx);
    viennacl::vector_base<double, size_t, ptrdiff_t> v2(y.get_ocl_mem(),
                     size_type(n), size_type(offY), difference_type(1), ctx);
    v2 += alpha * v1;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}
void OclDevice::axpby_double(const uint_tp n, const double alpha,
                   vptr<const double> x, const double beta, vptr<double> y) {
  this->scal_double(n, beta, y);
  this->axpy_double(n, alpha, x, y);
}
void OclDevice::scal_double(const uint_tp n, const double alpha,
                             vptr<double> x) {
  uint_tp offX = x.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x.get_ocl_mem(), true,
        CL_MAP_READ | CL_MAP_WRITE, sizeof(double) * offX, sizeof(double) * n, 0,
        NULL, NULL, NULL));

    caffe_scal<double>(n, alpha, xptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                            xptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(clblasDscal(n, alpha, x.get_ocl_mem(), offX,
            1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incx = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Scal<double>(
        n,
        alpha,
        x.get_ocl_mem(), offX, incx,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(x.get_ocl_mem(),
                      size_type(n), size_type(offX), difference_type(1), ctx);
    v1 *= alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}
void OclDevice::dot_double(const uint_tp n, vptr<const double> x,
                           vptr<const double> y, double* out) {
  uint_tp offX = x.get_ocl_off();
  uint_tp offY = y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* Xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offX, sizeof(double) * n, 0, NULL, NULL, NULL));
    double* Yptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), y.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offY, sizeof(double) * n, 0, NULL, NULL, NULL));

    *out = caffe_dot<double>(n, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                            Xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), y.get_ocl_mem(),
                            Yptr, 0, NULL, NULL);

  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err;
    cl_mem gpuout = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
        sizeof(double), NULL, &err);
    cl_mem scratch = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
        n * sizeof(double), NULL, &err);

    OPENCL_CL_BLAS_CHECK(
        clblasDdot(n, gpuout, 0, x.get_ocl_mem(), offX, 1, y.get_ocl_mem(),
            offY, 1, scratch, 1, &queue, 0, NULL, NULL));

    shared_ptr<ocl_dev_ptr<double> > oclptr_gpuout
                               = make_shared<ocl_dev_ptr<double> >(gpuout);
    vptr<double> vptr_gpuout(oclptr_gpuout);
    this->memcpy(sizeof(double), vptr<void>(vptr_gpuout), out);
    clReleaseMemObject(gpuout);
    clReleaseMemObject(scratch);

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err = CL_SUCCESS;
    cl_mem Z = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      sizeof(double), NULL, &err);

    const size_t offZ = 0;
    const size_t incX = 1;
    const size_t incY = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Dot<double>(
        n,
        Z, offZ,
        x.get_ocl_mem(), offX, incX,
        y.get_ocl_mem(), offY, incY,
        &queue));

    shared_ptr<ocl_dev_ptr<double> > oclptrZ
                              = make_shared<ocl_dev_ptr<double> >(Z, offZ);
    vptr<double> vptrZ(oclptrZ);
    this->memcpy(sizeof(double), vptr<void>(vptrZ), out);
    clReleaseMemObject(Z);

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(x.get_ocl_mem(),
                   size_type(n), size_type(offX), difference_type(1), ctx);
    viennacl::vector_base<double, size_t, ptrdiff_t> v2(y.get_ocl_mem(),
                   size_type(n), size_type(offY), difference_type(1), ctx);

    *out = viennacl::linalg::inner_prod(v1, v2);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}
void OclDevice::asum_double(const uint_tp n, vptr<const double> x, double* y) {
  uint_tp offX = x.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* Xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offX, sizeof(double) * n, 0, NULL, NULL, NULL));

    *y = caffe_asum<double>(n, Xptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                            Xptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err;
    cl_mem gpuout = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
        sizeof(double), NULL, &err);
    cl_mem scratch = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
        n * sizeof(double), NULL, &err);

    OPENCL_CL_BLAS_CHECK(
        clblasSasum(n, gpuout, 0, x.get_ocl_mem(), offX, 1,
            scratch, 1, &queue, 0, NULL, NULL));

    shared_ptr<ocl_dev_ptr<double> > oclptr_gpuout
                               = make_shared<ocl_dev_ptr<double> >(gpuout);
    vptr<double> vptr_gpuout(oclptr_gpuout);
    this->memcpy(sizeof(double), vptr<void>(vptr_gpuout), y);

    clReleaseMemObject(gpuout);
    clReleaseMemObject(scratch);

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err = CL_SUCCESS;
    cl_mem Z = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      sizeof(double), NULL, &err);

    const size_t offZ = 0;
    const size_t incX = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Asum<double>(
        n,
        Z, offZ,
        x.get_ocl_mem(), offX, incX,
        &queue));

    shared_ptr<ocl_dev_ptr<double> > oclptrZ
                      = make_shared<ocl_dev_ptr<double> >(Z, offZ);
    vptr<double> vptrZ(oclptrZ);
    this->memcpy(sizeof(double), vptr<void>(vptrZ), y);
    clReleaseMemObject(Z);

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(x.get_ocl_mem(),
                        size_type(n), size_type(offX), difference_type(1), ctx);

    *y = viennacl::linalg::norm_1(v1);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}
void OclDevice::scale_double(const uint_tp n, const double alpha,
                             vptr<const double> x, vptr<double> y) {
  uint_tp offX = x.get_ocl_off();
  uint_tp offY = y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* Xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offX, sizeof(double) * n, 0, NULL, NULL, NULL));
    double* Yptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), y.get_ocl_mem(), true, CL_MAP_WRITE,
        sizeof(double) * offY, sizeof(double) * n, 0, NULL, NULL, NULL));

    caffe_scale<double>(n, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                            Xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), y.get_ocl_mem(),
                            Yptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(
        clblasDcopy(n, x.get_ocl_mem(), offX, 1, y.get_ocl_mem(), offY, 1, 1,
                    &queue, 0, NULL, NULL));
    OPENCL_CL_BLAS_CHECK(
        clblasDscal(n, alpha, y.get_ocl_mem(), offY, 1, 1, &queue,
                    0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Copy<double>(
        n,
        x.get_ocl_mem(), offX, incX,
        y.get_ocl_mem(), offY, incY,
        &queue));
    OPENCL_CLBLAST_CHECK(
      clblast::Scal<double>(
        n,
        alpha,
        y.get_ocl_mem(), offY, incY,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(x.get_ocl_mem(),
                        size_type(n), size_type(offX), difference_type(1), ctx);
    viennacl::vector_base<double, size_t, ptrdiff_t> v2(y.get_ocl_mem(),
                        size_type(n), size_type(offY), difference_type(1), ctx);

    v2 = v1 * alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}
#endif  // USE_DOUBLE

}  // namespace caffe

#endif  // USE_OPENCL
