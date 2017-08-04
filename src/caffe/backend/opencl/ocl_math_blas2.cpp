#include "caffe/backend/opencl/ocl_math.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"

namespace caffe {

#ifdef USE_OPENCL

void ocl_device::gemv_half(const CBLAS_TRANSPOSE TransA, const uint_tp M,
                    const uint_tp N, const half_float::half alpha,
                    vptr<half_float::half> A,
                    vptr<half_float::half> x, const half_float::half beta,
                    vptr<half_float::half> y) {
#if defined(USE_GPU_HALF)
  uint_tp offA = A.get_ocl_off();
  uint_tp offx = x.get_ocl_off();
  uint_tp offy = y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

#if defined(USE_CLBLAS)
  clblasTranspose clTransA =
  (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;

  cl_command_queue queue = ctx.get_queue().handle().get();

  OPENCL_CL_BLAS_CHECK(
      clblasHgemv(clblasRowMajor,
          clTransA, M, N, alpha, A.get_ocl_mem(), offA, N, x.get_ocl_mem(),
          offx, 1, beta, y.get_ocl_mem(), offy, 1, 1, &queue, 0, NULL, NULL));
#elif defined(USE_CLBLAST)

  cl_command_queue queue = ctx.get_queue().handle().get();

  clblast::Layout layout = clblast::Layout::kRowMajor;
  clblast::Transpose a_transpose = (TransA == CblasNoTrans) ?
    clblast::Transpose::kNo : clblast::Transpose::kYes;

  const size_t ldA = N;
  const size_t incx = 1;
  const size_t incy = 1;

  OPENCL_CLBLAST_CHECK(
    clblast::Gemv<half_float::half>(
      layout, a_transpose,
      M, N,
      alpha,
      A.get_ocl_mem(), offA, ldA,
      x.get_ocl_mem(), offx, incx,
      beta,
      y.get_ocl_mem(), offy, incy,
      &queue));

#else  // default (ViennaCL)
    NOT_IMPLEMENTED;
#endif  // clBLAS, CLBlast, or default (ViennaCL)
#else  // USE_GPU_HALF
  NOT_IMPLEMENTED;
#endif // USE_GPU_HALF
}

void ocl_device::gemv_float(const CBLAS_TRANSPOSE TransA, const uint_tp M,
                    const uint_tp N, const float alpha,
                    vptr<float> A,
                    vptr<float> x, const float beta,
                    vptr<float> y) {
  uint_tp offA = A.get_ocl_off();
  uint_tp offx = x.get_ocl_off();
  uint_tp offy = y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Aptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), A.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offA, sizeof(float) * M * N, 0, NULL, NULL, NULL));
    float* xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offx, sizeof(float) * (TransA == CblasTrans) ? M : N, 0,
        NULL,
        NULL, NULL));
    float* yptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), y.get_ocl_mem(), true,
        CL_MAP_READ | CL_MAP_WRITE, sizeof(float) * offy,
        sizeof(float) * (TransA == CblasTrans) ? N : M, 0, NULL, NULL, NULL));

    caffe_cpu_gemv<float>(TransA, M, N, alpha, Aptr, xptr, beta, yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), A.get_ocl_mem(),
                            Aptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                            xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), y.get_ocl_mem(),
                            yptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)
    clblasTranspose clTransA =
    (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(
        clblasSgemv(clblasRowMajor,
            clTransA, M, N, alpha, A.get_ocl_mem(), offA, N, x.get_ocl_mem(),
            offx, 1, beta, y.get_ocl_mem(), offy, 1, 1, &queue, 0, NULL, NULL));
#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    clblast::Layout layout = clblast::Layout::kRowMajor;
    clblast::Transpose a_transpose = (TransA == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;

    const size_t ldA = N;
    const size_t incx = 1;
    const size_t incy = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Gemv<float>(
        layout, a_transpose,
        M, N,
        alpha,
        A.get_ocl_mem(), offA, ldA,
        x.get_ocl_mem(), offx, incx,
        beta,
        y.get_ocl_mem(), offy, incy,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(
        x.get_ocl_mem(), size_type((TransA == CblasTrans) ? M : N),
        size_type(offx), difference_type(1), ctx);
    viennacl::vector_base<float, size_t, ptrdiff_t> v2(
        y.get_ocl_mem(), size_type((TransA == CblasTrans) ? N : M),
        size_type(offy), difference_type(1), ctx);
    viennacl::matrix_base<float, size_t, ptrdiff_t> mat(
                                  A.get_ocl_mem(), ctx, size_type(M),
                                  size_type(0),
                                  difference_type(1),
                                  size_type(M),
                                  size_type(N),
                                  size_type(offA),
                                  difference_type(1),
                                  size_type(N)
                                  VCL_ROW_MAJOR);
    v2 *= beta;
    if (TransA == CblasTrans) {
      v2 += alpha * viennacl::linalg::prod(viennacl::trans(mat), v1);
    } else {
      v2 += alpha * viennacl::linalg::prod(mat, v1);
    }
#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

void ocl_device::gemv_double(const CBLAS_TRANSPOSE TransA, const uint_tp M,
                    const uint_tp N, const double alpha,
                    vptr<double> A,
                    vptr<double> x, const double beta,
                    vptr<double> y) {
  uint_tp offA = A.get_ocl_off();
  uint_tp offx = x.get_ocl_off();
  uint_tp offy = y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* Aptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), A.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offA, sizeof(double) * M * N, 0, NULL, NULL, NULL));
    double* xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offx, sizeof(double) * (TransA == CblasTrans) ? M : N,
            0, NULL, NULL, NULL));
    double* yptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), y.get_ocl_mem(), true,
        CL_MAP_READ | CL_MAP_WRITE, sizeof(double) * offy,
        sizeof(double) * (TransA == CblasTrans) ? N : M, 0, NULL, NULL, NULL));

    caffe_cpu_gemv<double>(TransA, M, N, alpha, Aptr, xptr, beta, yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), A.get_ocl_mem(),
                            Aptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                            xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), y.get_ocl_mem(),
                            yptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)
    clblasTranspose clTransA =
    (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(
        clblasDgemv(clblasRowMajor,
            clTransA, M, N, alpha, A.get_ocl_mem(), offA, N, x.get_ocl_mem(),
            offx, 1, beta, y.get_ocl_mem(), offy, 1, 1, &queue, 0, NULL, NULL));
#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    clblast::Layout layout = clblast::Layout::kRowMajor;
    clblast::Transpose a_transpose = (TransA == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;

    const size_t ldA = N;
    const size_t incx = 1;
    const size_t incy = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Gemv<double>(
        layout, a_transpose,
        M, N,
        alpha,
        A.get_ocl_mem(), offA, ldA,
        x.get_ocl_mem(), offx, incx,
        beta,
        y.get_ocl_mem(), offy, incy,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(
        x.get_ocl_mem(), size_type((TransA == CblasTrans) ? M : N),
        size_type(offx), difference_type(1), ctx);
    viennacl::vector_base<double, size_t, ptrdiff_t> v2(
        y.get_ocl_mem(), size_type((TransA == CblasTrans) ? N : M),
        size_type(offy), difference_type(1), ctx);
    viennacl::matrix_base<double, size_t, ptrdiff_t> mat(
                                  A.get_ocl_mem(), ctx, size_type(M),
                                  size_type(0),
                                  difference_type(1),
                                  size_type(M),
                                  size_type(N),
                                  size_type(offA),
                                  difference_type(1),
                                  size_type(N)
                                  VCL_ROW_MAJOR);
    v2 *= beta;
    if (TransA == CblasTrans) {
      v2 += alpha * viennacl::linalg::prod(viennacl::trans(mat), v1);
    } else {
      v2 += alpha * viennacl::linalg::prod(mat, v1);
    }
#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}


#endif  // USE_OPENCL

}
