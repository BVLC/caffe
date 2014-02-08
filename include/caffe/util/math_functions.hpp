// Copyright 2013 Yangqing Jia

#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

//#include <mkl.h>
#include <eigen3/Eigen/Dense>

namespace caffe {

// Operations on aligned memory are faster than on unaligned memory.
// But unfortunately, the pointers passed in are not always aligned.
// Therefore, the memory-aligned Eigen::Map objects that wrap them
// cannot be assigned to. This happens in lrn_layer and makes
// test_lrn_layer crash with segmentation fault.
// TODO: Use aligned Eigen::Map when the pointer to be wrapped is aligned.

// Though the default map option is unaligned, making it explicit is no harm.
//const int data_alignment = Eigen::Aligned; // how is data allocated ?
const int data_alignment = Eigen::Unaligned;
typedef Eigen::Map<const Eigen::VectorXf, data_alignment> const_map_vector_float_t;
typedef Eigen::Map<Eigen::VectorXf, data_alignment> map_vector_float_t;
typedef Eigen::Map<const Eigen::VectorXd, data_alignment> const_map_vector_double_t;
typedef Eigen::Map<Eigen::VectorXd, data_alignment> map_vector_double_t;

// The default in Eigen is column-major. This is also the case if one
// of the convenience typedefs (Matrix3f, ArrayXXd, etc.) is used.
// http://eigen.tuxfamily.org/dox-devel/group__TopicStorageOrders.html
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXf;
typedef Eigen::Map<MatXf, data_alignment> map_matrix_float_t;
typedef Eigen::Map<const MatXf, data_alignment> const_map_matrix_float_t;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXd;
typedef Eigen::Map<MatXd, data_alignment> map_matrix_double_t;
typedef Eigen::Map<const MatXd, data_alignment> const_map_matrix_double_t;

// From cblas.h
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

// Decaf gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void caffe_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

template <typename Dtype>
void caffe_gpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

template <typename Dtype>
void caffe_copy(const int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void caffe_gpu_copy(const int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void caffe_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_sqr(const int N, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b);

template <typename Dtype>
void caffe_vRngUniform(const int n, Dtype* r, const Dtype a, const Dtype b);

template <typename Dtype>
void caffe_vRngGaussian(const int n, Dtype* r, const Dtype a,
    const Dtype sigma);

template <typename Dtype>
void caffe_vRngBernoulli(const int n, Dtype* r, const double p);

template <typename Dtype>
void caffe_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y);

template <typename Dtype>
void caffe_gpu_dot(const int n, const Dtype* x, const Dtype* y, Dtype* out);

}  // namespace caffe


#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
