#ifdef USE_OPENCL

#include "caffe/backend/opencl/ocl_math.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"

namespace caffe {

#ifdef USE_HALF
void OclDevice::gemv_half(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                          const uint_tp n, const half_fp alpha,
                          vptr<const half_fp> a,
                          vptr<const half_fp> x,
                          const half_fp beta,
                          vptr<half_fp> y,
                          const QuantizerValues* const alpha_quant,
                          const QuantizerValues* const a_quant,
                          const QuantizerValues* const x_quant,
                          const QuantizerValues* const beta_quant,
                          const QuantizerValues* const y_quant) {
  uint_tp offA = a.get_ocl_off();
  uint_tp offx = x.get_ocl_off();
  uint_tp offy = y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

#if defined(USE_CLBLAS)
  clblasTranspose clTransA =
  (trans_a == CblasNoTrans) ? clblasNoTrans : clblasTrans;

  cl_command_queue queue = ctx.get_queue().handle().get();

  OPENCL_CL_BLAS_CHECK(
      clblasHgemv(clblasRowMajor,
          clTransA, m, n, alpha, a.get_ocl_mem(), offA, n, x.get_ocl_mem(),
          offx, 1, beta, y.get_ocl_mem(), offy, 1, 1, &queue, 0, NULL, NULL));
#elif defined(USE_CLBLAST)

  cl_command_queue queue = ctx.get_queue().handle().get();

  clblast::Layout layout = clblast::Layout::kRowMajor;
  clblast::Transpose a_transpose = (trans_a == CblasNoTrans) ?
    clblast::Transpose::kNo : clblast::Transpose::kYes;

  const size_t ldA = n;
  const size_t incx = 1;
  const size_t incy = 1;

  OPENCL_CLBLAST_CHECK(
    clblast::Gemv<clblast::half>(
      layout, a_transpose,
      m, n,
      clblast::FloatToHalf(alpha),
      a.get_ocl_mem(), offA, ldA,
      x.get_ocl_mem(), offx, incx,
      clblast::FloatToHalf(beta),
      y.get_ocl_mem(), offy, incy,
      &queue));

#else  // default (ViennaCL)
    Device::gemv_half(trans_a, m, n, alpha, a, x, beta, y);
#endif  // clBLAS, CLBlast, or default (ViennaCL)
}
#endif  // USE_HALF


#ifdef USE_SINGLE
void OclDevice::gemv_float(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                           const uint_tp n, const float alpha,
                           vptr<const float> a,
                           vptr<const float> x, const float beta,
                           vptr<float> y,
                           const QuantizerValues* const alpha_quant,
                           const QuantizerValues* const a_quant,
                           const QuantizerValues* const x_quant,
                           const QuantizerValues* const beta_quant,
                           const QuantizerValues* const y_quant) {
  uint_tp offA = a.get_ocl_off();
  uint_tp offx = x.get_ocl_off();
  uint_tp offy = y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Aptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), a.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offA, sizeof(float) * m * n, 0, NULL, NULL, NULL));
    float* xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offx, sizeof(float) * ((trans_a == CblasTrans) ? m : n),
        0, NULL, NULL, NULL));
    float* yptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), y.get_ocl_mem(), true,
        CL_MAP_READ | CL_MAP_WRITE, sizeof(float) * offy,
        sizeof(float) * ((trans_a == CblasTrans) ? n : m),
        0, NULL, NULL, NULL));

    caffe_gemv<float>(trans_a, m, n, alpha, Aptr, xptr, beta, yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), a.get_ocl_mem(),
                            Aptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                            xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), y.get_ocl_mem(),
                            yptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)
    clblasTranspose clTransA =
    (trans_a == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(
        clblasSgemv(clblasRowMajor,
            clTransA, m, n, alpha, a.get_ocl_mem(), offA, n, x.get_ocl_mem(),
            offx, 1, beta, y.get_ocl_mem(), offy, 1, 1, &queue, 0, NULL, NULL));
#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    clblast::Layout layout = clblast::Layout::kRowMajor;
    clblast::Transpose a_transpose = (trans_a == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;

    const size_t ldA = n;
    const size_t incx = 1;
    const size_t incy = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Gemv<float>(
        layout, a_transpose,
        m, n,
        alpha,
        a.get_ocl_mem(), offA, ldA,
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
        x.get_ocl_mem(), size_type((trans_a == CblasTrans) ? m : n),
        size_type(offx), difference_type(1), ctx);
    viennacl::vector_base<float, size_t, ptrdiff_t> v2(
        y.get_ocl_mem(), size_type((trans_a == CblasTrans) ? n : m),
        size_type(offy), difference_type(1), ctx);
    viennacl::matrix_base<float, size_t, ptrdiff_t> mat(
                                  a.get_ocl_mem(), ctx, size_type(m),
                                  size_type(0),
                                  difference_type(1),
                                  size_type(m),
                                  size_type(n),
                                  size_type(offA),
                                  difference_type(1),
                                  size_type(n)
                                  VCL_ROW_MAJOR);
    v2 *= beta;
    if (trans_a == CblasTrans) {
      v2 += alpha * viennacl::linalg::prod(viennacl::trans(mat), v1);
    } else {
      v2 += alpha * viennacl::linalg::prod(mat, v1);
    }
#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}
#endif  // USE_SINGLE


#ifdef USE_DOUBLE
void OclDevice::gemv_double(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                            const uint_tp n, const double alpha,
                            vptr<const double> a,
                            vptr<const double> x, const double beta,
                            vptr<double> y,
                            const QuantizerValues* const alpha_quant,
                            const QuantizerValues* const a_quant,
                            const QuantizerValues* const x_quant,
                            const QuantizerValues* const beta_quant,
                            const QuantizerValues* const y_quant) {
  uint_tp offA = a.get_ocl_off();
  uint_tp offx = x.get_ocl_off();
  uint_tp offy = y.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* Aptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), a.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offA, sizeof(double) * m * n, 0, NULL, NULL, NULL));
    double* xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offx,
        sizeof(double) * ((trans_a == CblasTrans) ? m : n),
        0, NULL, NULL, NULL));
    double* yptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), y.get_ocl_mem(), true,
        CL_MAP_READ | CL_MAP_WRITE, sizeof(double) * offy,
        sizeof(double) * ((trans_a == CblasTrans) ? n : m),
        0, NULL, NULL, NULL));

    caffe_gemv<double>(trans_a, m, n, alpha, Aptr, xptr, beta, yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), a.get_ocl_mem(),
                            Aptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                            xptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), y.get_ocl_mem(),
                            yptr, 0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)
    clblasTranspose clTransA =
    (trans_a == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();

    OPENCL_CL_BLAS_CHECK(
        clblasDgemv(clblasRowMajor,
            clTransA, m, n, alpha, a.get_ocl_mem(), offA, n, x.get_ocl_mem(),
            offx, 1, beta, y.get_ocl_mem(), offy, 1, 1, &queue, 0, NULL, NULL));
#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    clblast::Layout layout = clblast::Layout::kRowMajor;
    clblast::Transpose a_transpose = (trans_a == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;

    const size_t ldA = n;
    const size_t incx = 1;
    const size_t incy = 1;

    OPENCL_CLBLAST_CHECK(
      clblast::Gemv<double>(
        layout, a_transpose,
        m, n,
        alpha,
        a.get_ocl_mem(), offA, ldA,
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
        x.get_ocl_mem(), size_type((trans_a == CblasTrans) ? m : n),
        size_type(offx), difference_type(1), ctx);
    viennacl::vector_base<double, size_t, ptrdiff_t> v2(
        y.get_ocl_mem(), size_type((trans_a == CblasTrans) ? n : m),
        size_type(offy), difference_type(1), ctx);
    viennacl::matrix_base<double, size_t, ptrdiff_t> mat(
                                  a.get_ocl_mem(), ctx, size_type(m),
                                  size_type(0),
                                  difference_type(1),
                                  size_type(m),
                                  size_type(n),
                                  size_type(offA),
                                  difference_type(1),
                                  size_type(n)
                                  VCL_ROW_MAJOR);
    v2 *= beta;
    if (trans_a == CblasTrans) {
      v2 += alpha * viennacl::linalg::prod(viennacl::trans(mat), v1);
    } else {
      v2 += alpha * viennacl::linalg::prod(mat, v1);
    }
#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}
#endif  // USE_DOUBLE

}  // namespace caffe

#endif  // USE_OPENCL
