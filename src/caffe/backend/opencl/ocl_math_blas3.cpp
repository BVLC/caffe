#include "caffe/backend/opencl/ocl_math.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"

namespace caffe {

#ifdef USE_OPENCL

void OclDevice::gemm_half(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b,
                            const uint_tp m, const uint_tp n, const uint_tp k,
                            const half_float::half alpha,
                            vptr<const half_float::half> a,
                            vptr<const half_float::half> b,
                            const half_float::half beta,
                            vptr<half_float::half> c) {
#if defined(USE_GPU_HALF)
  uint_tp offA = a.get_ocl_off();
  uint_tp offB = b.get_ocl_off();
  uint_tp offC = c.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  int_tp lda = (trans_a == CblasNoTrans) ? k : m;
  int_tp ldb = (trans_b == CblasNoTrans) ? n : k;
  int_tp ldc = n;
#if defined(USE_CLBLAS)
  clblasOrder clOrder = clblasRowMajor;
  clblasTranspose clTransA =
  (trans_a == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  clblasTranspose clTransB =
  (trans_b == CblasNoTrans) ? clblasNoTrans : clblasTrans;

  cl_command_queue queue = ctx.get_queue().handle().get();
  OPENCL_CL_BLAS_CHECK(
      clblasHgemm(clOrder, clTransA, clTransB,
          m, n, k, alpha, a.get_ocl_mem(), offA, lda,
          b.get_ocl_mem(), offB, ldb, beta,
          c.get_ocl_mem(), offC, ldc, 1, &queue, 0, NULL, NULL));
#elif defined(USE_CLBLAST)

  cl_command_queue queue = ctx.get_queue().handle().get();

  clblast::Layout layout = clblast::Layout::kRowMajor;
  clblast::Transpose a_transpose = (trans_a == CblasNoTrans) ?
    clblast::Transpose::kNo : clblast::Transpose::kYes;
  clblast::Transpose b_transpose = (trans_b == CblasNoTrans) ?
    clblast::Transpose::kNo : clblast::Transpose::kYes;

  OPENCL_CLBLAST_CHECK(
    clblast::Gemm<half_float::half>(
      layout, a_transpose, b_transpose,
      m, n, k,
      alpha,
      a.get_ocl_mem(), offA, lda,
      b.get_ocl_mem(), offB, ldb,
      beta,
      c.get_ocl_mem(), offC, ldc,
      &queue));
#else  // default (ViennaCL)
  NOT_IMPLEMENTED;
#endif  // clBLAS, CLBlast, or default (ViennaCL)
#else  // USE_GPU_HALF
  NOT_IMPLEMENTED;
#endif // USE_GPU_HALF
}

void OclDevice::gemm_float(const CBLAS_TRANSPOSE trans_a,
                           const CBLAS_TRANSPOSE trans_b,
                           const uint_tp m, const uint_tp n, const uint_tp k,
                           const float alpha, vptr<const float> a,
                           vptr<const float> b, const float beta,
                           vptr<float> c) {
  uint_tp offA = a.get_ocl_off();
  uint_tp offB = b.get_ocl_off();
  uint_tp offC = c.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Aptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), a.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offA, sizeof(float) * m * k, 0, NULL, NULL, NULL));
    float* Bptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), b.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offB, sizeof(float) * n * k, 0, NULL, NULL, NULL));
    float* Cptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), c.get_ocl_mem(), true,
        CL_MAP_READ | CL_MAP_WRITE,
        sizeof(float) * offC, sizeof(float) * m * n, 0, NULL, NULL, NULL));

    caffe_cpu_gemm<float>(trans_a, trans_b, m, n, k, alpha, Aptr, Bptr, beta,
                          Cptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), a.get_ocl_mem(),
                            Aptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), b.get_ocl_mem(),
                            Bptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), c.get_ocl_mem(),
                            Cptr, 0, NULL, NULL);
  } else {
    int_tp lda = (trans_a == CblasNoTrans) ? k : m;
    int_tp ldb = (trans_b == CblasNoTrans) ? n : k;
    int_tp ldc = n;
#if defined(USE_CLBLAS)
    clblasOrder clOrder = clblasRowMajor;
    clblasTranspose clTransA =
    (trans_a == CblasNoTrans) ? clblasNoTrans : clblasTrans;
    clblasTranspose clTransB =
    (trans_b == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();
    OPENCL_CL_BLAS_CHECK(
        clblasSgemm(clOrder, clTransA, clTransB,
            m, n, k, alpha, a.get_ocl_mem(), offA, lda,
            b.get_ocl_mem(), offB, ldb, beta,
            c.get_ocl_mem(), offC, ldc, 1, &queue, 0, NULL, NULL));
#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    clblast::Layout layout = clblast::Layout::kRowMajor;
    clblast::Transpose a_transpose = (trans_a == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;
    clblast::Transpose b_transpose = (trans_b == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;

    OPENCL_CLBLAST_CHECK(
      clblast::Gemm<float>(
        layout, a_transpose, b_transpose,
        m, n, k,
        alpha,
        a.get_ocl_mem(), offA, lda,
        b.get_ocl_mem(), offB, ldb,
        beta,
        c.get_ocl_mem(), offC, ldc,
        &queue));
#else  // default (ViennaCL)

    typedef typename viennacl::matrix_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::matrix_base<float,
        uint_tp, int_tp>::size_type difference_type;

    size_type A_size1 = static_cast<size_type>((trans_a == CblasTrans) ? k : m);
    size_type A_size2 = static_cast<size_type>((trans_a == CblasTrans) ? m : k);

    size_type B_size1 = static_cast<size_type>((trans_b == CblasTrans) ? n : k);
    size_type B_size2 = static_cast<size_type>((trans_b == CblasTrans) ? k : n);

    viennacl::matrix_base<float, size_t, ptrdiff_t> matA(a.get_ocl_mem(),
                                                       ctx, A_size1,
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(m), A_size2,
                                                       size_type(offA),
                                                       difference_type(1),
                                                       size_type(lda)
                                                       VCL_ROW_MAJOR);

    viennacl::matrix_base<float, size_t, ptrdiff_t> matB(b.get_ocl_mem(),
                                                       ctx, B_size1,
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(k), B_size2,
                                                       size_type(offB),
                                                       difference_type(1),
                                                       size_type(ldb)
                                                       VCL_ROW_MAJOR);

    viennacl::matrix_base<float, size_t, ptrdiff_t> matC(c.get_ocl_mem(),
                                                       ctx, size_type(m),
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(m),
                                                       size_type(n),
                                                       size_type(offC),
                                                       difference_type(1),
                                                       size_type(ldc)
                                                       VCL_ROW_MAJOR);

    if (trans_a == CblasTrans && trans_b == CblasTrans)
      viennacl::linalg::prod_impl(viennacl::trans(matA), viennacl::trans(matB),
                                  matC, alpha, beta);
    else if (trans_a == CblasTrans && trans_b == CblasNoTrans)
      viennacl::linalg::prod_impl(viennacl::trans(matA), matB, matC, alpha,
                                  beta);
    else if (trans_a == CblasNoTrans && trans_b == CblasTrans)
      viennacl::linalg::prod_impl(matA, viennacl::trans(matB), matC, alpha,
                                  beta);
    else if (trans_a == CblasNoTrans && trans_b == CblasNoTrans)
      viennacl::linalg::prod_impl(matA, matB, matC, alpha, beta);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

void OclDevice::gemm_double(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b,
                            const uint_tp m, const uint_tp n, const uint_tp k,
                            const double alpha, vptr<const double> a,
                            vptr<const double> b,
                            const double beta, vptr<double> c) {
  uint_tp offA = a.get_ocl_off();
  uint_tp offB = b.get_ocl_off();
  uint_tp offC = c.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* Aptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), a.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offA, sizeof(double) * m * k, 0, NULL, NULL, NULL));
    double* Bptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), b.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offB, sizeof(double) * n * k, 0, NULL, NULL, NULL));
    double* Cptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), c.get_ocl_mem(), true,
        CL_MAP_READ | CL_MAP_WRITE,
        sizeof(double) * offC, sizeof(double) * m * n, 0, NULL, NULL, NULL));

    caffe_cpu_gemm<double>(trans_a, trans_b, m, n, k, alpha, Aptr, Bptr, beta,
                          Cptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), a.get_ocl_mem(),
                            Aptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), b.get_ocl_mem(),
                            Bptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), c.get_ocl_mem(),
                            Cptr, 0, NULL, NULL);
  } else {
    int_tp lda = (trans_a == CblasNoTrans) ? k : m;
    int_tp ldb = (trans_b == CblasNoTrans) ? n : k;
    int_tp ldc = n;
#if defined(USE_CLBLAS)
    clblasOrder clOrder = clblasRowMajor;
    clblasTranspose clTransA =
    (trans_a == CblasNoTrans) ? clblasNoTrans : clblasTrans;
    clblasTranspose clTransB =
    (trans_b == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();
    OPENCL_CL_BLAS_CHECK(
        clblasDgemm(clOrder, clTransA, clTransB,
            m, n, k, alpha, a.get_ocl_mem(), offA, lda,
            b.get_ocl_mem(), offB, ldb, beta,
            c.get_ocl_mem(), offC, ldc, 1, &queue, 0, NULL, NULL));
#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    clblast::Layout layout = clblast::Layout::kRowMajor;
    clblast::Transpose a_transpose = (trans_a == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;
    clblast::Transpose b_transpose = (trans_b == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;

    OPENCL_CLBLAST_CHECK(
      clblast::Gemm<double>(
        layout, a_transpose, b_transpose,
        m, n, k,
        alpha,
        a.get_ocl_mem(), offA, lda,
        b.get_ocl_mem(), offB, ldb,
        beta,
        c.get_ocl_mem(), offC, ldc,
        &queue));
#else  // default (ViennaCL)

    typedef typename viennacl::matrix_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::matrix_base<double,
        uint_tp, int_tp>::size_type difference_type;

    size_type A_size1 = static_cast<size_type>((trans_a == CblasTrans) ? k : m);
    size_type A_size2 = static_cast<size_type>((trans_a == CblasTrans) ? m : k);

    size_type B_size1 = static_cast<size_type>((trans_b == CblasTrans) ? n : k);
    size_type B_size2 = static_cast<size_type>((trans_b == CblasTrans) ? k : n);

    viennacl::matrix_base<double, size_t, ptrdiff_t> matA(a.get_ocl_mem(),
                                                       ctx, A_size1,
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(m), A_size2,
                                                       size_type(offA),
                                                       difference_type(1),
                                                       size_type(lda)
                                                       VCL_ROW_MAJOR);

    viennacl::matrix_base<double, size_t, ptrdiff_t> matB(b.get_ocl_mem(),
                                                       ctx, B_size1,
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(k), B_size2,
                                                       size_type(offB),
                                                       difference_type(1),
                                                       size_type(ldb)
                                                       VCL_ROW_MAJOR);

    viennacl::matrix_base<double, size_t, ptrdiff_t> matC(c.get_ocl_mem(),
                                                       ctx, size_type(m),
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(m),
                                                       size_type(n),
                                                       size_type(offC),
                                                       difference_type(1),
                                                       size_type(ldc)
                                                       VCL_ROW_MAJOR);

    if (trans_a == CblasTrans && trans_b == CblasTrans)
      viennacl::linalg::prod_impl(viennacl::trans(matA), viennacl::trans(matB),
                                  matC, alpha, beta);
    else if (trans_a == CblasTrans && trans_b == CblasNoTrans)
      viennacl::linalg::prod_impl(viennacl::trans(matA), matB, matC, alpha,
                                  beta);
    else if (trans_a == CblasNoTrans && trans_b == CblasTrans)
      viennacl::linalg::prod_impl(matA, viennacl::trans(matB), matC, alpha,
                                  beta);
    else if (trans_a == CblasNoTrans && trans_b == CblasNoTrans)
      viennacl::linalg::prod_impl(matA, matB, matC, alpha, beta);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

#endif  // USE_OPENCL

}  // namespace caffe
