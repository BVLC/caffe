#include "caffe/backend/opencl/ocl_math.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"

namespace caffe {

#ifdef USE_OPENCL

void ocl_device::gemm_half(const CBLAS_TRANSPOSE TransA,
                            const CBLAS_TRANSPOSE TransB,
                            const uint_tp M, const uint_tp N, const uint_tp K,
                            const half_float::half alpha,
                            vptr<half_float::half> A,
                            vptr<half_float::half> B,
                            const half_float::half beta,
                            vptr<half_float::half> C) {
#if defined(USE_GPU_HALF)
  uint_tp offA = A.get_ocl_off();
  uint_tp offB = B.get_ocl_off();
  uint_tp offC = C.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  int_tp lda = (TransA == CblasNoTrans) ? K : M;
  int_tp ldb = (TransB == CblasNoTrans) ? N : K;
  int_tp ldc = N;
#if defined(USE_CLBLAS)
  clblasOrder clOrder = clblasRowMajor;
  clblasTranspose clTransA =
  (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  clblasTranspose clTransB =
  (TransB == CblasNoTrans) ? clblasNoTrans : clblasTrans;

  cl_command_queue queue = ctx.get_queue().handle().get();
  OPENCL_CL_BLAS_CHECK(
      clblasHgemm(clOrder, clTransA, clTransB,
          M, N, K, alpha, A.get_ocl_mem(), offA, lda,
          B.get_ocl_mem(), offB, ldb, beta,
          C.get_ocl_mem(), offC, ldc, 1, &queue, 0, NULL, NULL));
#elif defined(USE_CLBLAST)

  cl_command_queue queue = ctx.get_queue().handle().get();

  clblast::Layout layout = clblast::Layout::kRowMajor;
  clblast::Transpose a_transpose = (TransA == CblasNoTrans) ?
    clblast::Transpose::kNo : clblast::Transpose::kYes;
  clblast::Transpose b_transpose = (TransB == CblasNoTrans) ?
    clblast::Transpose::kNo : clblast::Transpose::kYes;

  OPENCL_CLBLAST_CHECK(
    clblast::Gemm<half_float::half>(
      layout, a_transpose, b_transpose,
      M, N, K,
      alpha,
      A.get_ocl_mem(), offA, lda,
      B.get_ocl_mem(), offB, ldb,
      beta,
      C.get_ocl_mem(), offC, ldc,
      &queue));
#else  // default (ViennaCL)
  NOT_IMPLEMENTED;
#endif  // clBLAS, CLBlast, or default (ViennaCL)
#else  // USE_GPU_HALF
  NOT_IMPLEMENTED;
#endif // USE_GPU_HALF
}

void ocl_device::gemm_float(const CBLAS_TRANSPOSE TransA,
                            const CBLAS_TRANSPOSE TransB,
                            const uint_tp M, const uint_tp N, const uint_tp K,
                            const float alpha, vptr<float> A,
                            vptr<float> B,
                            const float beta, vptr<float> C) {
  uint_tp offA = A.get_ocl_off();
  uint_tp offB = B.get_ocl_off();
  uint_tp offC = C.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Aptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), A.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offA, sizeof(float) * M * K, 0, NULL, NULL, NULL));
    float* Bptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), B.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(float) * offB, sizeof(float) * N * K, 0, NULL, NULL, NULL));
    float* Cptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), C.get_ocl_mem(), true,
        CL_MAP_READ | CL_MAP_WRITE,
        sizeof(float) * offC, sizeof(float) * M * N, 0, NULL, NULL, NULL));

    caffe_cpu_gemm<float>(TransA, TransB, M, N, K, alpha, Aptr, Bptr, beta,
                          Cptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), A.get_ocl_mem(),
                            Aptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), B.get_ocl_mem(),
                            Bptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), C.get_ocl_mem(),
                            Cptr, 0, NULL, NULL);
  } else {
    int_tp lda = (TransA == CblasNoTrans) ? K : M;
    int_tp ldb = (TransB == CblasNoTrans) ? N : K;
    int_tp ldc = N;
#if defined(USE_CLBLAS)
    clblasOrder clOrder = clblasRowMajor;
    clblasTranspose clTransA =
    (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
    clblasTranspose clTransB =
    (TransB == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();
    OPENCL_CL_BLAS_CHECK(
        clblasSgemm(clOrder, clTransA, clTransB,
            M, N, K, alpha, A.get_ocl_mem(), offA, lda,
            B.get_ocl_mem(), offB, ldb, beta,
            C.get_ocl_mem(), offC, ldc, 1, &queue, 0, NULL, NULL));
#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    clblast::Layout layout = clblast::Layout::kRowMajor;
    clblast::Transpose a_transpose = (TransA == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;
    clblast::Transpose b_transpose = (TransB == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;

    OPENCL_CLBLAST_CHECK(
      clblast::Gemm<float>(
        layout, a_transpose, b_transpose,
        M, N, K,
        alpha,
        A.get_ocl_mem(), offA, lda,
        B.get_ocl_mem(), offB, ldb,
        beta,
        C.get_ocl_mem(), offC, ldc,
        &queue));
#else  // default (ViennaCL)

    typedef typename viennacl::matrix_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::matrix_base<float,
        uint_tp, int_tp>::size_type difference_type;

    size_type A_size1 = static_cast<size_type>((TransA == CblasTrans) ? K : M);
    size_type A_size2 = static_cast<size_type>((TransA == CblasTrans) ? M : K);

    size_type B_size1 = static_cast<size_type>((TransB == CblasTrans) ? N : K);
    size_type B_size2 = static_cast<size_type>((TransB == CblasTrans) ? K : N);

    viennacl::matrix_base<float, size_t, ptrdiff_t> matA(A.get_ocl_mem(),
                                                       ctx, A_size1,
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(M), A_size2,
                                                       size_type(offA),
                                                       difference_type(1),
                                                       size_type(lda)
                                                       VCL_ROW_MAJOR);

    viennacl::matrix_base<float, size_t, ptrdiff_t> matB(B.get_ocl_mem(),
                                                       ctx, B_size1,
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(K), B_size2,
                                                       size_type(offB),
                                                       difference_type(1),
                                                       size_type(ldb)
                                                       VCL_ROW_MAJOR);

    viennacl::matrix_base<float, size_t, ptrdiff_t> matC(C.get_ocl_mem(),
                                                       ctx, size_type(M),
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(M),
                                                       size_type(N),
                                                       size_type(offC),
                                                       difference_type(1),
                                                       size_type(ldc)
                                                       VCL_ROW_MAJOR);

    if (TransA == CblasTrans && TransB == CblasTrans)
      viennacl::linalg::prod_impl(viennacl::trans(matA), viennacl::trans(matB),
                                  matC, alpha, beta);
    else if (TransA == CblasTrans && TransB == CblasNoTrans)
      viennacl::linalg::prod_impl(viennacl::trans(matA), matB, matC, alpha,
                                  beta);
    else if (TransA == CblasNoTrans && TransB == CblasTrans)
      viennacl::linalg::prod_impl(matA, viennacl::trans(matB), matC, alpha,
                                  beta);
    else if (TransA == CblasNoTrans && TransB == CblasNoTrans)
      viennacl::linalg::prod_impl(matA, matB, matC, alpha, beta);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

void ocl_device::gemm_double(const CBLAS_TRANSPOSE TransA,
                            const CBLAS_TRANSPOSE TransB,
                            const uint_tp M, const uint_tp N, const uint_tp K,
                            const double alpha, vptr<double> A,
                            vptr<double> B,
                            const double beta, vptr<double> C) {
  uint_tp offA = A.get_ocl_off();
  uint_tp offB = B.get_ocl_off();
  uint_tp offC = C.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* Aptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), A.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offA, sizeof(double) * M * K, 0, NULL, NULL, NULL));
    double* Bptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), B.get_ocl_mem(), true, CL_MAP_READ,
        sizeof(double) * offB, sizeof(double) * N * K, 0, NULL, NULL, NULL));
    double* Cptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), C.get_ocl_mem(), true,
        CL_MAP_READ | CL_MAP_WRITE,
        sizeof(double) * offC, sizeof(double) * M * N, 0, NULL, NULL, NULL));

    caffe_cpu_gemm<double>(TransA, TransB, M, N, K, alpha, Aptr, Bptr, beta,
                          Cptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), A.get_ocl_mem(),
                            Aptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), B.get_ocl_mem(),
                            Bptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), C.get_ocl_mem(),
                            Cptr, 0, NULL, NULL);
  } else {
    int_tp lda = (TransA == CblasNoTrans) ? K : M;
    int_tp ldb = (TransB == CblasNoTrans) ? N : K;
    int_tp ldc = N;
#if defined(USE_CLBLAS)
    clblasOrder clOrder = clblasRowMajor;
    clblasTranspose clTransA =
    (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
    clblasTranspose clTransB =
    (TransB == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();
    OPENCL_CL_BLAS_CHECK(
        clblasDgemm(clOrder, clTransA, clTransB,
            M, N, K, alpha, A.get_ocl_mem(), offA, lda,
            B.get_ocl_mem(), offB, ldb, beta,
            C.get_ocl_mem(), offC, ldc, 1, &queue, 0, NULL, NULL));
#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    clblast::Layout layout = clblast::Layout::kRowMajor;
    clblast::Transpose a_transpose = (TransA == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;
    clblast::Transpose b_transpose = (TransB == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;

    OPENCL_CLBLAST_CHECK(
      clblast::Gemm<double>(
        layout, a_transpose, b_transpose,
        M, N, K,
        alpha,
        A.get_ocl_mem(), offA, lda,
        B.get_ocl_mem(), offB, ldb,
        beta,
        C.get_ocl_mem(), offC, ldc,
        &queue));
#else  // default (ViennaCL)

    typedef typename viennacl::matrix_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::matrix_base<double,
        uint_tp, int_tp>::size_type difference_type;

    size_type A_size1 = static_cast<size_type>((TransA == CblasTrans) ? K : M);
    size_type A_size2 = static_cast<size_type>((TransA == CblasTrans) ? M : K);

    size_type B_size1 = static_cast<size_type>((TransB == CblasTrans) ? N : K);
    size_type B_size2 = static_cast<size_type>((TransB == CblasTrans) ? K : N);

    viennacl::matrix_base<double, size_t, ptrdiff_t> matA(A.get_ocl_mem(),
                                                       ctx, A_size1,
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(M), A_size2,
                                                       size_type(offA),
                                                       difference_type(1),
                                                       size_type(lda)
                                                       VCL_ROW_MAJOR);

    viennacl::matrix_base<double, size_t, ptrdiff_t> matB(B.get_ocl_mem(),
                                                       ctx, B_size1,
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(K), B_size2,
                                                       size_type(offB),
                                                       difference_type(1),
                                                       size_type(ldb)
                                                       VCL_ROW_MAJOR);

    viennacl::matrix_base<double, size_t, ptrdiff_t> matC(C.get_ocl_mem(),
                                                       ctx, size_type(M),
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(M),
                                                       size_type(N),
                                                       size_type(offC),
                                                       difference_type(1),
                                                       size_type(ldc)
                                                       VCL_ROW_MAJOR);

    if (TransA == CblasTrans && TransB == CblasTrans)
      viennacl::linalg::prod_impl(viennacl::trans(matA), viennacl::trans(matB),
                                  matC, alpha, beta);
    else if (TransA == CblasTrans && TransB == CblasNoTrans)
      viennacl::linalg::prod_impl(viennacl::trans(matA), matB, matC, alpha,
                                  beta);
    else if (TransA == CblasNoTrans && TransB == CblasTrans)
      viennacl::linalg::prod_impl(matA, viennacl::trans(matB), matC, alpha,
                                  beta);
    else if (TransA == CblasNoTrans && TransB == CblasNoTrans)
      viennacl::linalg::prod_impl(matA, matB, matC, alpha, beta);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

#endif  // USE_OPENCL

}  // namespace caffe
