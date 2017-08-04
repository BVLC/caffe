#include "caffe/backend/opencl/ocl_math.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"

namespace caffe {

#ifdef USE_OPENCL

void ocl_device::axpy_half(const uint_tp N, const half_float::half alpha,
                        vptr<half_float::half> X, vptr<half_float::half> Y) {
#if defined(USE_GPU_HALF)
  uint_tp offX = X.get_ocl_off();
  uint_tp offY = Y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

#if defined(USE_CLBLAS)

  cl_command_queue queue = ctx.get_queue().handle().get();

  OPENCL_CL_BLAS_CHECK(
      clblasHaxpy(N, alpha, X.get_ocl_mem(), offX,
          1, Y.get_ocl_mem(), offY, 1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

  cl_command_queue queue = ctx.get_queue().handle().get();

  const size_t incX = 1;
  const size_t incY = 1;

  OPENCL_CLBLAST_CHECK(
    clblast::Axpy<half_float::half>(
      N,
      alpha,
      X.get_ocl_mem(), offX, incX,
      Y.get_ocl_mem(), offY, incY,
      &queue));

#else  // default (ViennaCL)
  NOT_IMPLEMENTED;
#endif  // clBLAS, CLBlast, or default (ViennaCL)
#else  // USE_GPU_HALF
  NOT_IMPLEMENTED;
#endif  // USE_GPU_HALF
}

void ocl_device::axpy_float(const uint_tp N, const float alpha,
                        vptr<float> X, vptr<float> Y) {

  uint_tp offX = X.get_ocl_off();
  uint_tp offY = Y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offX, sizeof(float) * N, 0, NULL, NULL, NULL));
    float* Yptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Y.get_ocl_mem(), true, CL_MAP_WRITE,
        sizeof(float) * offY, sizeof(float) * N, 0, NULL, NULL, NULL));

    caffe_axpy<float>(N, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X.get_ocl_mem(),
                            Xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Y.get_ocl_mem(),
                            Yptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(
        clblasSaxpy(N, alpha, X.get_ocl_mem(), offX,
            1, Y.get_ocl_mem(), offY, 1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Axpy<float>(
        N,
        alpha,
        X.get_ocl_mem(), offX, incX,
        Y.get_ocl_mem(), offY, incY,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(X.get_ocl_mem(),
                     size_type(N), size_type(offX), difference_type(1), ctx);
    viennacl::vector_base<float, size_t, ptrdiff_t> v2(Y.get_ocl_mem(),
                     size_type(N), size_type(offY), difference_type(1), ctx);
    v2 += alpha * v1;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

void ocl_device::axpy_double(const uint_tp N, const double alpha,
                        vptr<double> X, vptr<double> Y) {
  uint_tp offX = X.get_ocl_off();
  uint_tp offY = Y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* Xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offX, sizeof(double) * N, 0, NULL, NULL, NULL));
    double* Yptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Y.get_ocl_mem(), true, CL_MAP_WRITE,
        sizeof(double) * offY, sizeof(double) * N, 0, NULL, NULL, NULL));

    caffe_axpy<double>(N, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X.get_ocl_mem(),
                            Xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Y.get_ocl_mem(),
                            Yptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(
        clblasDaxpy(N, alpha, X.get_ocl_mem(), offX,
            1, Y.get_ocl_mem(), offY, 1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Axpy<double>(
        N,
        alpha,
        X.get_ocl_mem(), offX, incX,
        Y.get_ocl_mem(), offY, incY,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(X.get_ocl_mem(),
                     size_type(N), size_type(offX), difference_type(1), ctx);
    viennacl::vector_base<double, size_t, ptrdiff_t> v2(Y.get_ocl_mem(),
                     size_type(N), size_type(offY), difference_type(1), ctx);
    v2 += alpha * v1;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

void ocl_device::axpby_half(const uint_tp N, const half_float::half alpha,
                   vptr<half_float::half> X,
                   const half_float::half beta, vptr<half_float::half> Y) {
  this->scal_half(N, beta, Y);
  this->axpy_half(N, alpha, X, Y);
}

void ocl_device::axpby_float(const uint_tp N, const float alpha,
                   vptr<float> X, const float beta, vptr<float> Y) {
  this->scal_float(N, beta, Y);
  this->axpy_float(N, alpha, X, Y);
}

void ocl_device::axpby_double(const uint_tp N, const double alpha,
                   vptr<double> X, const double beta, vptr<double> Y) {
  this->scal_double(N, beta, Y);
  this->axpy_double(N, alpha, X, Y);
}


void ocl_device::scal_half(const uint_tp N, const half_float::half alpha,
                  vptr<half_float::half> X) {
#ifdef USE_GPU_HALF
  uint_tp offX = X.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

#if defined(USE_CLBLAS)

  cl_command_queue queue = ctx.get_queue().handle().get();

  OPENCL_CL_BLAS_CHECK(clblasHscal(N, alpha, X.get_ocl_mem(), offX,
          1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

  cl_command_queue queue = ctx.get_queue().handle().get();

  const size_t incx = 1;

  OPENCL_CLBLAST_CHECK(
    clblast::Scal<half_float::half>(
      N,
      alpha,
      X.get_ocl_mem(), offX, incx,
      &queue));

#else  // default (ViennaCL)
  NOT_IMPLEMENTED;
#endif  // clBLAS, CLBlast, or default (ViennaCL)
#else  // USE_GPU_HALF
  NOT_IMPLEMENTED;
#endif
}

void ocl_device::scal_float(const uint_tp N, const float alpha, vptr<float> X) {
  uint_tp offX = X.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X.get_ocl_mem(), true,
        CL_MAP_READ | CL_MAP_WRITE, sizeof(float) * offX, sizeof(float) * N, 0,
        NULL, NULL, NULL));

    caffe_scal<float>(N, alpha, xptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X.get_ocl_mem(),
                            xptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(clblasSscal(N, alpha, X.get_ocl_mem(), offX,
            1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incx = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Scal<float>(
        N,
        alpha,
        X.get_ocl_mem(), offX, incx,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(X.get_ocl_mem(),
                      size_type(N), size_type(offX), difference_type(1), ctx);
    v1 *= alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

void ocl_device::scal_double(const uint_tp N, const double alpha,
                             vptr<double> X) {
  uint_tp offX = X.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X.get_ocl_mem(), true,
        CL_MAP_READ | CL_MAP_WRITE, sizeof(double) * offX, sizeof(double) * N, 0,
        NULL, NULL, NULL));

    caffe_scal<double>(N, alpha, xptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X.get_ocl_mem(),
                            xptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(clblasDscal(N, alpha, X.get_ocl_mem(), offX,
            1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incx = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Scal<double>(
        N,
        alpha,
        X.get_ocl_mem(), offX, incx,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(X.get_ocl_mem(),
                      size_type(N), size_type(offX), difference_type(1), ctx);
    v1 *= alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

void ocl_device::dot_half(const uint_tp n, vptr<half_float::half> X,
                          vptr<half_float::half> Y, half_float::half* out) {
#ifdef USE_GPU_HALF
  uint_tp offX = X.get_ocl_off();
  uint_tp offY = Y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

#if defined(USE_CLBLAS)

  cl_command_queue queue = ctx.get_queue().handle().get();

  cl_int err;
  cl_mem gpuout = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      sizeof(half_float::half), NULL, &err);
  cl_mem scratch = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      n * sizeof(half_float::half), NULL, &err);

  OPENCL_CL_BLAS_CHECK(
      clblasHdot(n, gpuout, 0, X.get_ocl_mem(), offX, 1, Y.get_ocl_mem(),
          offY, 1, scratch, 1, &queue, 0, NULL, NULL));

  ocl_device::memcpy(sizeof(half_float::half), gpuout, 0, out, &ctx);

  clReleaseMemObject(gpuout);
  clReleaseMemObject(scratch);

#elif defined(USE_CLBLAST)

  cl_command_queue queue = ctx.get_queue().handle().get();

  cl_int err = CL_SUCCESS;
  cl_mem Z = clCreateBuffer(ctx.handle().get(),
       vptr<half_float::half>_READ_WRITE, sizeof(half_float::half), NULL, &err);

  const size_t offZ = 0;
  const size_t incX = 1;
  const size_t incY = 1;

  OPENCL_CLBLAST_CHECK(
    clblast::Dot<half_float::half>(
      n,
      Z, offZ,
      X.get_ocl_mem(), offX, incX,
      Y.get_ocl_mem(), offY, incY,
      &queue));

  ocl_device::memcpy(sizeof(half_float::half), Z, offZ, out, &ctx);
  clReleaseMemObject(Z);

#else  // default (ViennaCL)
  NOT_IMPLEMENTED;
#endif  // clBLAS, CLBlast, or default (ViennaCL)
#else  // USE_GPU_HALF
  NOT_IMPLEMENTED;
#endif  // USE_GPU_HALF
}

void ocl_device::dot_float(const uint_tp n, vptr<float> X, vptr<float> Y,
                           float* out) {

  uint_tp offX = X.get_ocl_off();
  uint_tp offY = Y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offX, sizeof(float) * n, 0, NULL, NULL, NULL));
    float* Yptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Y.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offY, sizeof(float) * n, 0, NULL, NULL, NULL));

    *out = caffe_cpu_dot<float>(n, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X.get_ocl_mem(),
                            Xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Y.get_ocl_mem(),
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
        clblasSdot(n, gpuout, 0, X.get_ocl_mem(), offX, 1, Y.get_ocl_mem(),
            offY, 1, scratch, 1, &queue, 0, NULL, NULL));

    ocl_device::memcpy(sizeof(float), gpuout, 0, out, &ctx);

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
        X.get_ocl_mem(), offX, incX,
        Y.get_ocl_mem(), offY, incY,
        &queue));

    ocl_device::memcpy(sizeof(float), Z, offZ, out, &ctx);
    clReleaseMemObject(Z);

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(X.get_ocl_mem(),
                   size_type(n), size_type(offX), difference_type(1), ctx);
    viennacl::vector_base<float, size_t, ptrdiff_t> v2(Y.get_ocl_mem(),
                   size_type(n), size_type(offY), difference_type(1), ctx);

    *out = viennacl::linalg::inner_prod(v1, v2);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

void ocl_device::dot_double(const uint_tp n, vptr<double> X, vptr<double> Y,
                            double* out) {

  uint_tp offX = X.get_ocl_off();
  uint_tp offY = Y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offX, sizeof(double) * n, 0, NULL, NULL, NULL));
    double* Yptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Y.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offY, sizeof(double) * n, 0, NULL, NULL, NULL));

    *out = caffe_cpu_dot<double>(n, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X.get_ocl_mem(),
                            Xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Y.get_ocl_mem(),
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
        clblasDdot(n, gpuout, 0, X.get_ocl_mem(), offX, 1, Y.get_ocl_mem(),
            offY, 1, scratch, 1, &queue, 0, NULL, NULL));

    ocl_device::memcpy(sizeof(double), gpuout, 0, out, &ctx);

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
        X.get_ocl_mem(), offX, incX,
        Y.get_ocl_mem(), offY, incY,
        &queue));

    ocl_device::memcpy(sizeof(double), Z, offZ, out, &ctx);
    clReleaseMemObject(Z);

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(X.get_ocl_mem(),
                   size_type(n), size_type(offX), difference_type(1), ctx);
    viennacl::vector_base<double, size_t, ptrdiff_t> v2(Y.get_ocl_mem(),
                   size_type(n), size_type(offY), difference_type(1), ctx);

    *out = viennacl::linalg::inner_prod(v1, v2);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

void ocl_device::asum_half(const uint_tp n, vptr<half_float::half> X,
                           half_float::half* Y) {
#ifdef USE_GPU_HALF
  uint_tp offX = X.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

#if defined(USE_CLBLAS)

  cl_command_queue queue = ctx.get_queue().handle().get();

  cl_int err;
  cl_mem gpuout = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      sizeof(half_float::half), NULL, &err);
  cl_mem scratch = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      n * sizeof(half_float::half), NULL, &err);

  OPENCL_CL_BLAS_CHECK(
      clblasHasum(n, gpuout, 0, X.get_ocl_mem(), offX, 1,
          scratch, 1, &queue, 0, NULL, NULL));

  ocl_device::memcpy(sizeof(half_float::half), gpuout, 0, Y, &ctx);

  clReleaseMemObject(gpuout);
  clReleaseMemObject(scratch);

#elif defined(USE_CLBLAST)

  cl_command_queue queue = ctx.get_queue().handle().get();

  cl_int err = CL_SUCCESS;
  cl_mem Z = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
    sizeof(half_float::half), NULL, &err);

  const size_t offZ = 0;
  const size_t incX = 1;

  OPENCL_CLBLAST_CHECK(
    clblast::Asum<half_float::half>(
      n,
      Z, offZ,
      X.get_ocl_mem(), offX, incX,
      &queue));

  ocl_device::memcpy(sizeof(half_float::half), Z, offZ, Y, &ctx);

  clReleaseMemObject(Z);

#else  // default (ViennaCL)
  NOT_IMPLEMENTED;
#endif  // clBLAS, CLBlast, or default (ViennaCL)
#else  // USE_GPU_HALF
  NOT_IMPLEMENTED;
#endif  // USE_GPU_HALF
}

void ocl_device::asum_float(const uint_tp n, vptr<float> X, float* Y) {
  uint_tp offX = X.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offX, sizeof(float) * n, 0, NULL, NULL, NULL));

    *Y = caffe_cpu_asum<float>(n, Xptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X.get_ocl_mem(),
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
        half_float::halfsSasum(n, gpuout, 0, X.get_ocl_mem(), offX, 1,
            scratch, 1, &queue, 0, NULL, NULL));

    ocl_device::memcpy(sizeof(float), gpuout, 0, Y, &ctx);

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
        X.get_ocl_mem(), offX, incX,
        &queue));

    ocl_device::memcpy(sizeof(float), Z, offZ, Y, &ctx);

    clReleaseMemObject(Z);

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(X.get_ocl_mem(),
                        size_type(n), size_type(offX), difference_type(1), ctx);

    *Y = viennacl::linalg::norm_1(v1);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}


void ocl_device::asum_double(const uint_tp n, vptr<double> X, double* Y) {
  uint_tp offX = X.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* Xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offX, sizeof(double) * n, 0, NULL, NULL, NULL));

    *Y = caffe_cpu_asum<double>(n, Xptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X.get_ocl_mem(),
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
        clblasSasum(n, gpuout, 0, X.get_ocl_mem(), offX, 1,
            scratch, 1, &queue, 0, NULL, NULL));

    ocl_device::memcpy(sizeof(double), gpuout, 0, Y, &ctx);

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
        X.get_ocl_mem(), offX, incX,
        &queue));

    ocl_device::memcpy(sizeof(double), Z, offZ, Y, &ctx);

    clReleaseMemObject(Z);

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(X.get_ocl_mem(),
                        size_type(n), size_type(offX), difference_type(1), ctx);

    *Y = viennacl::linalg::norm_1(v1);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}


void ocl_device::scale_half(const uint_tp n, const half_float::half alpha,
                           vptr<half_float::half> X, vptr<half_float::half> Y) {
  uint_tp offX = X.get_ocl_off();
  uint_tp offY = Y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    half_float::half* Xptr = reinterpret_cast<half_float::half*>(
        clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(half_float::half) * offX, sizeof(half_float::half) * n, 0,
        NULL, NULL, NULL));
    half_float::half* Yptr = reinterpret_cast<half_float::half*>(
        clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Y.get_ocl_mem(), true, CL_MAP_WRITE,
        sizeof(half_float::half) * offY, sizeof(half_float::half) * n, 0,
        NULL, NULL, NULL));

    caffe_cpu_scale<half_float::half>(n, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X.get_ocl_mem(),
                            Xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Y.get_ocl_mem(),
                            Yptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(
        clblasHcopy(n, X.get_ocl_mem(), offX, 1, Y.get_ocl_mem(), offY, 1, 1,
                    &queue, 0, NULL, NULL));
    OPENCL_CL_BLAS_CHECK(
        clblasHscal(n, alpha, Y.get_ocl_mem(), offY, 1, 1, &queue,
                    0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Copy<half_float::half>(
        n,
        X.get_ocl_mem(), offX, incX,
        Y.get_ocl_mem(), offY, incY,
        &queue));
    OPENCL_CLBLAST_CHECK(
      clblast::Scal<half_float::half>(
        n,
        alpha,
        Y.get_ocl_mem(), offY, incY,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<half_float::half,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<half_float::half,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<half_float::half, size_t, ptrdiff_t> v1(
                     X.get_ocl_mem(), size_type(n), size_type(offX),
                     difference_type(1), ctx);
    viennacl::vector_base<half_float::half, size_t, ptrdiff_t> v2(
                     Y.get_ocl_mem(), size_type(n), size_type(offY),
                     difference_type(1), ctx);

    v2 = v1 * alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

void ocl_device::scale_float(const uint_tp n, const float alpha,
                             vptr<float> X, vptr<float> Y) {
  uint_tp offX = X.get_ocl_off();
  uint_tp offY = Y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offX, sizeof(float) * n, 0, NULL, NULL, NULL));
    float* Yptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Y.get_ocl_mem(), true, CL_MAP_WRITE,
        sizeof(float) * offY, sizeof(float) * n, 0, NULL, NULL, NULL));

    caffe_cpu_scale<float>(n, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X.get_ocl_mem(),
                            Xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Y.get_ocl_mem(),
                            Yptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(
        clblasScopy(n, X.get_ocl_mem(), offX, 1, Y.get_ocl_mem(), offY, 1, 1,
                    &queue, 0, NULL, NULL));
    OPENCL_CL_BLAS_CHECK(
        clblasSscal(n, alpha, Y.get_ocl_mem(), offY, 1, 1, &queue,
                    0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Copy<float>(
        n,
        X.get_ocl_mem(), offX, incX,
        Y.get_ocl_mem(), offY, incY,
        &queue));
    OPENCL_CLBLAST_CHECK(
      clblast::Scal<float>(
        n,
        alpha,
        Y.get_ocl_mem(), offY, incY,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(X.get_ocl_mem(),
                        size_type(n), size_type(offX), difference_type(1), ctx);
    viennacl::vector_base<float, size_t, ptrdiff_t> v2(Y.get_ocl_mem(),
                        size_type(n), size_type(offY), difference_type(1), ctx);

    v2 = v1 * alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

void ocl_device::scale_double(const uint_tp n, const double alpha,
                             vptr<double> X, vptr<double> Y) {
  uint_tp offX = X.get_ocl_off();
  uint_tp offY = Y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* Xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offX, sizeof(double) * n, 0, NULL, NULL, NULL));
    double* Yptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Y.get_ocl_mem(), true, CL_MAP_WRITE,
        sizeof(double) * offY, sizeof(double) * n, 0, NULL, NULL, NULL));

    caffe_cpu_scale<double>(n, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X.get_ocl_mem(),
                            Xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Y.get_ocl_mem(),
                            Yptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(
        clblasDcopy(n, X.get_ocl_mem(), offX, 1, Y.get_ocl_mem(), offY, 1, 1,
                    &queue, 0, NULL, NULL));
    OPENCL_CL_BLAS_CHECK(
        clblasDscal(n, alpha, Y.get_ocl_mem(), offY, 1, 1, &queue,
                    0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Copy<double>(
        n,
        X.get_ocl_mem(), offX, incX,
        Y.get_ocl_mem(), offY, incY,
        &queue));
    OPENCL_CLBLAST_CHECK(
      clblast::Scal<double>(
        n,
        alpha,
        Y.get_ocl_mem(), offY, incY,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(X.get_ocl_mem(),
                        size_type(n), size_type(offX), difference_type(1), ctx);
    viennacl::vector_base<double, size_t, ptrdiff_t> v2(Y.get_ocl_mem(),
                        size_type(n), size_type(offY), difference_type(1), ctx);

    v2 = v1 * alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

#endif  // USE_OPENCL

}
