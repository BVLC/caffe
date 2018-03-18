#ifdef USE_OPENCL
#include "caffe/backend/opencl/ocl_math.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"

namespace caffe {

#ifdef USE_HALF
void OclDevice::gemm_half(const CBLAS_TRANSPOSE trans_A,
                            const CBLAS_TRANSPOSE trans_B,
                            const uint_tp M, const uint_tp N, const uint_tp K,
                            const half_fp alpha,
                            vptr<const half_fp> A,
                            vptr<const half_fp> B,
                            const half_fp beta,
                            vptr<half_fp> C,
                            const QuantizerValues* const alpha_quant,
                            const QuantizerValues* const a_quant,
                            const QuantizerValues* const b_quant,
                            const QuantizerValues* const beta_quant,
                            const QuantizerValues* const c_quant) {
  uint_tp offA = A.get_ocl_off();
  uint_tp offB = B.get_ocl_off();
  uint_tp offC = C.get_ocl_off();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());

  int_tp lda = (trans_A == CblasNoTrans) ? K : M;
  int_tp ldb = (trans_B == CblasNoTrans) ? N : K;
  int_tp ldc = N;
#if defined(USE_CLBLAS)
  clblasOrder clOrder = clblasRowMajor;
  clblasTranspose clTransA =
  (trans_A == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  clblasTranspose clTransB =
  (trans_B == CblasNoTrans) ? clblasNoTrans : clblasTrans;

  cl_command_queue queue = ctx.get_queue().handle().get();
  OPENCL_CL_BLAS_CHECK(
      clblasHgemm(clOrder, clTransA, clTransB,
          M, N, K, alpha, A.get_ocl_mem(), offA, lda,
          B.get_ocl_mem(), offB, ldb, beta,
          C.get_ocl_mem(), offC, ldc, 1, &queue, 0, NULL, NULL));
#elif defined(USE_CLBLAST)
  cl_command_queue queue = ctx.get_queue().handle().get();

  clblast::Layout layout = clblast::Layout::kRowMajor;
  clblast::Transpose a_transpose = (trans_A == CblasNoTrans) ?
    clblast::Transpose::kNo : clblast::Transpose::kYes;
  clblast::Transpose b_transpose = (trans_B == CblasNoTrans) ?
    clblast::Transpose::kNo : clblast::Transpose::kYes;

  OPENCL_CLBLAST_CHECK(
    clblast::Gemm<clblast::half>(
      layout, a_transpose, b_transpose,
      M, N, K,
      clblast::FloatToHalf(alpha),
      A.get_ocl_mem(), offA, lda,
      B.get_ocl_mem(), offB, ldb,
      clblast::FloatToHalf(beta),
      C.get_ocl_mem(), offC, ldc,
      &queue));
#else  // default (ViennaCL)
  Device::gemm_half(trans_A, trans_B, M, N, K, alpha, A, B, beta, C,
                    alpha_quant, a_quant, b_quant, beta_quant, c_quant);
#endif  // clBLAS, CLBlast, or default (ViennaCL)
}
#endif  // USE_HALF


#ifdef USE_SINGLE
void OclDevice::gemm_float(const CBLAS_TRANSPOSE trans_A,
                           const CBLAS_TRANSPOSE trans_B,
                           const uint_tp M, const uint_tp N, const uint_tp K,
                           const float alpha, vptr<const float> A,
                           vptr<const float> B, const float beta,
                           vptr<float> C,
                           const QuantizerValues* const alpha_quant,
                           const QuantizerValues* const a_quant,
                           const QuantizerValues* const b_quant,
                           const QuantizerValues* const beta_quant,
                           const QuantizerValues* const c_quant) {
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

    caffe_gemm<float>(trans_A, trans_B, M, N, K, alpha, Aptr, Bptr, beta,
                      Cptr, alpha_quant, a_quant, b_quant, beta_quant,
                      c_quant);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), A.get_ocl_mem(),
                            Aptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), B.get_ocl_mem(),
                            Bptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), C.get_ocl_mem(),
                            Cptr, 0, NULL, NULL);
  } else {
    int_tp lda = (trans_A == CblasNoTrans) ? K : M;
    int_tp ldb = (trans_B == CblasNoTrans) ? N : K;
    int_tp ldc = N;
#if defined(USE_CLBLAS)
    clblasOrder clOrder = clblasRowMajor;
    clblasTranspose clTransA =
    (trans_A == CblasNoTrans) ? clblasNoTrans : clblasTrans;
    clblasTranspose clTransB =
    (trans_B == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();
    OPENCL_CL_BLAS_CHECK(
        clblasSgemm(clOrder, clTransA, clTransB,
            M, N, K, alpha, A.get_ocl_mem(), offA, lda,
            B.get_ocl_mem(), offB, ldb, beta,
            C.get_ocl_mem(), offC, ldc, 1, &queue, 0, NULL, NULL));
#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    clblast::Layout layout = clblast::Layout::kRowMajor;
    clblast::Transpose a_transpose = (trans_A == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;
    clblast::Transpose b_transpose = (trans_B == CblasNoTrans) ?
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

    size_type A_size1 = static_cast<size_type>((trans_A == CblasTrans) ? K : M);
    size_type A_size2 = static_cast<size_type>((trans_A == CblasTrans) ? M : K);

    size_type B_size1 = static_cast<size_type>((trans_B == CblasTrans) ? N : K);
    size_type B_size2 = static_cast<size_type>((trans_B == CblasTrans) ? K : N);

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

    if (trans_A == CblasTrans && trans_B == CblasTrans)
      viennacl::linalg::prod_impl(viennacl::trans(matA), viennacl::trans(matB),
                                  matC, alpha, beta);
    else if (trans_A == CblasTrans && trans_B == CblasNoTrans)
      viennacl::linalg::prod_impl(viennacl::trans(matA), matB, matC, alpha,
                                  beta);
    else if (trans_A == CblasNoTrans && trans_B == CblasTrans)
      viennacl::linalg::prod_impl(matA, viennacl::trans(matB), matC, alpha,
                                  beta);
    else if (trans_A == CblasNoTrans && trans_B == CblasNoTrans)
      viennacl::linalg::prod_impl(matA, matB, matC, alpha, beta);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}
#endif  // USE_SINGLE


#ifdef USE_DOUBLE
void OclDevice::gemm_double(const CBLAS_TRANSPOSE trans_A,
                            const CBLAS_TRANSPOSE trans_B,
                            const uint_tp M, const uint_tp N, const uint_tp K,
                            const double alpha, vptr<const double> A,
                            vptr<const double> B,
                            const double beta, vptr<double> C,
                            const QuantizerValues* const a_quant,
                            const QuantizerValues* const b_quant,
                            const QuantizerValues* const c_quant) {
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

    caffe_gemm<double>(trans_A, trans_B, M, N, K, alpha, Aptr, Bptr, beta,
                          Cptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), A.get_ocl_mem(),
                            Aptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), B.get_ocl_mem(),
                            Bptr, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), C.get_ocl_mem(),
                            Cptr, 0, NULL, NULL);
  } else {
    int_tp lda = (trans_A == CblasNoTrans) ? K : M;
    int_tp ldb = (trans_B == CblasNoTrans) ? N : K;
    int_tp ldc = N;
#if defined(USE_CLBLAS)
    clblasOrder clOrder = clblasRowMajor;
    clblasTranspose clTransA =
    (trans_A == CblasNoTrans) ? clblasNoTrans : clblasTrans;
    clblasTranspose clTransB =
    (trans_B == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();
    OPENCL_CL_BLAS_CHECK(
        clblasDgemm(clOrder, clTransA, clTransB,
            M, N, K, alpha, A.get_ocl_mem(), offA, lda,
            B.get_ocl_mem(), offB, ldb, beta,
            C.get_ocl_mem(), offC, ldc, 1, &queue, 0, NULL, NULL));
#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    clblast::Layout layout = clblast::Layout::kRowMajor;
    clblast::Transpose a_transpose = (trans_A == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;
    clblast::Transpose b_transpose = (trans_B == CblasNoTrans) ?
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

    size_type A_size1 = static_cast<size_type>((trans_A == CblasTrans) ? K : M);
    size_type A_size2 = static_cast<size_type>((trans_A == CblasTrans) ? M : K);

    size_type B_size1 = static_cast<size_type>((trans_B == CblasTrans) ? N : K);
    size_type B_size2 = static_cast<size_type>((trans_B == CblasTrans) ? K : N);

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

    if (trans_A == CblasTrans && trans_B == CblasTrans)
      viennacl::linalg::prod_impl(viennacl::trans(matA), viennacl::trans(matB),
                                  matC, alpha, beta);
    else if (trans_A == CblasTrans && trans_B == CblasNoTrans)
      viennacl::linalg::prod_impl(viennacl::trans(matA), matB, matC, alpha,
                                  beta);
    else if (trans_A == CblasNoTrans && trans_B == CblasTrans)
      viennacl::linalg::prod_impl(matA, viennacl::trans(matB), matC, alpha,
                                  beta);
    else if (trans_A == CblasNoTrans && trans_B == CblasNoTrans)
      viennacl::linalg::prod_impl(matA, matB, matC, alpha, beta);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}
#endif  // USE_DOUBLE

}  // namespace caffe
#endif  // USE_OPENCL
