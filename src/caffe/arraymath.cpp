#include "caffe/arraymath.hpp"
#include <boost/make_shared.hpp>
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

using boost::make_shared;
namespace caffe {

namespace arraymath_detail {
template<typename T>
const T *getData(const Array<T> *a, ArrayMode m = AR_DEFAULT) {
  if (m == AR_DEFAULT)
    m = a->effectiveMode();
  if (m == AR_CPU)
    return a->cpu_data();
  return a->gpu_data();
}
template<typename T>
T *getMutableData(Array<T> *a, ArrayMode m = AR_DEFAULT) {
  if (m == AR_DEFAULT)
    m = a->effectiveMode();
  if (m == AR_CPU)
    return a->mutable_cpu_data();
  return a->mutable_gpu_data();
}
}  // namespace arraymath_detail

// Use openmp if supported
#if defined(_OPENMP)
#define OMP_PARALLEL_FOR _Pragma("omp parallel for")
#else
#define OMP_PARALLEL_FOR
#endif

/****** ArrayBase Pointer ******/
// Use this function to avoid having to store a reference to array base in the
// UnaryAE and BinaryAE
namespace arraymath_detail {
template<typename T> struct ArrayBasePointer {
  // Store either a memory reference or expression
  ArrayShape s_;
  ArrayMode m_;
  shared_ptr<SyncedMemory> memory_;
  shared_ptr<typename Expression<T>::Implementation> expression_;
  explicit ArrayBasePointer(const ArrayBase<T> & a) {
    try {
      // Try to fetch the memory
      memory_ = dynamic_cast<const Array<T>&>(a).memory();
      s_ = dynamic_cast<const Array<T>&>(a).shape();
      m_ = dynamic_cast<const Array<T>&>(a).mode();
    } catch( std::bad_cast ) {
      // Fetch the expression
      expression_ = dynamic_cast<const Expression<T>&>(a).imp();
    }
  }
  Array<T> eval() const {
    if (memory_) return Array<T>(memory_, s_, m_);
    return expression_->eval();
  }
};
}  // namespace arraymath_detail

/****** Unary array operations ******/
namespace arraymath_detail {
template<typename T>
class UnaryAE: public Expression<T>::Implementation {
 public:
  typedef void(*Func)(int, const T *, T *);
 protected:
  ArrayBasePointer<T> a_;
  Func f_;
 public:
  UnaryAE(Func f, const ArrayBase<T> &a, const ArrayMode &m)
    : Expression<T>::Implementation(a.shape(), m), a_(a), f_(f) {
    CHECK_NE(m, AR_DEFAULT) << "Array mode cannot be default!";
  }
  virtual void evaluate(Array<T> *target) const {
    int N = count(this->shape_);
    Array<T> a = a_.eval();
    f_(N, getData(&a, this->mode()), getMutableData(target, this->mode()));
  }
};
}  // namespace arraymath_detail

// Declare a GPU function
#ifdef CPU_ONLY
#define UNARY_GPU_DECLARE(name)
#else
#define UNARY_GPU_DECLARE(name)\
namespace arraymath_detail {\
template<typename T> void name##_gpu(int N, const T *a, T *r);\
}  // namespace arraymath_detail
#endif

// Define a CPU function
#define UNARY_CPU_DEFINE(name) namespace arraymath_detail {\
template<typename T> void name##_cpu(int N, const T *a, T *r) {\
  OMP_PARALLEL_FOR\
  for (int i = 0; i < N; i++)\
    r[i] = name(a[i]);\
}\
}  // namespace arraymath_detail

// Call a GPU function if available
#ifdef CPU_ONLY
#define UNARY_GPU_CALL(name)
#else
#define UNARY_GPU_CALL(name) if (a.effectiveMode() == AR_GPU)\
  return Expression<T>(make_shared<UnaryAE<T> >(&name##_gpu<T>, a, AR_GPU))
#endif

// Implement unary expressions
#define IMPLEMENT_UNARY(name) UNARY_GPU_DECLARE(name)\
UNARY_CPU_DEFINE(name)\
template<typename T>\
Expression<T> ARMath<T>::name(const ArrayBase<T> & a) {\
  using namespace arraymath_detail; /* NOLINT */\
  UNARY_GPU_CALL(name);\
  return Expression<T>(make_shared<UnaryAE<T> >(&name##_cpu<T>, a, AR_CPU));\
}

namespace arraymath_detail {
  template<typename T> T negate(const T &v) { return -v; }
  template<typename T> T sign(const T &v) { return v < 0 ? -1 : v > 0 ? 1 : 0; }
};

IMPLEMENT_UNARY(abs);
IMPLEMENT_UNARY(exp);
IMPLEMENT_UNARY(log);
IMPLEMENT_UNARY(negate);
IMPLEMENT_UNARY(sign);
IMPLEMENT_UNARY(sqrt);
#undef IMPLEMENT_UNARY
#undef UNARY_GPU_CALL
#undef UNARY_CPU_DEFINE
#undef UNARY_GPU_DECLARE



/****** Binary array operations ******/
namespace arraymath_detail {
template<typename T, typename T1, typename T2> class BinaryAE { /*NOT IMPL.*/ };

template<typename T>
class BinaryAE<T, ArrayBase<T>, ArrayBase<T> >:
  public Expression<T>::Implementation {
 public:
  typedef void(*Func)(int, const T *, const T *, T *);
 protected:
  ArrayBasePointer<T> a_, b_;
  Func f_;
 public:
  BinaryAE(Func f, const ArrayBase<T> &a, const ArrayBase<T> &b,
           const ArrayMode &m)
    : Expression<T>::Implementation(a.shape(), m), a_(a), b_(b), f_(f) {
    CHECK_NE(m, AR_DEFAULT) << "Array mode cannot be default!";
  }
  virtual void evaluate(Array<T> *target) const {
    size_t N = count(this->shape_);
    Array<T> a = a_.eval(), b = b_.eval();
    f_(N, getData(&a, this->mode()), getData(&b, this->mode()),
       getMutableData(target, this->mode()));
  }
};
template<typename T>
class BinaryAE<T, T, ArrayBase<T> >: public Expression<T>::Implementation {
 public:
  typedef void(*Func)(int, const T &, const T *, T *);
 protected:
  T a_;
  ArrayBasePointer<T> b_;
  Func f_;
 public:
  BinaryAE(Func f, const T &a, const ArrayBase<T> &b, const ArrayMode &m)
    : Expression<T>::Implementation(b.shape(), m), a_(a), b_(b), f_(f) {
    CHECK_NE(m, AR_DEFAULT) << "Array mode cannot be default!";
  }
  virtual void evaluate(Array<T> *target) const {
    size_t N = count(this->shape_);
    Array<T> b = b_.eval();
    f_(N, a_, getData(&b, this->mode()), getMutableData(target, this->mode()));
  }
};
template<typename T>
class BinaryAE<T, ArrayBase<T>, T>: public Expression<T>::Implementation {
 public:
  typedef void(*Func)(int, const T *, const T &, T *);
 protected:
  ArrayBasePointer<T> a_;
  T b_;
  Func f_;
 public:
  BinaryAE(Func f, const ArrayBase<T> &a, const T &b, const ArrayMode &m)
    : Expression<T>::Implementation(a.shape(), m), a_(a), b_(b), f_(f) {
    CHECK_NE(m, AR_DEFAULT) << "Array mode cannot be default!";
  }
  virtual void evaluate(Array<T> *target) const {
    size_t N = count(this->shape_);
    Array<T> a = a_.eval();
    f_(N, getData(&a, this->mode()), b_, getMutableData(target, this->mode()));
  }
};


template<typename T, typename T1, typename T2>
shared_ptr<typename Expression<T>::Implementation> makeBinaryAE( typename BinaryAE<T,T1,T2>::Func f, const T1 &a, const T2 &b, const ArrayMode &m) {  // NOLINT : Go buy a larger sceeen!
  return make_shared<BinaryAE<T, T1, T2> >(f, a, b, m);
}
}  // namespace arraymath_detail

// Declare a GPU function
#ifdef CPU_ONLY
#define BINARY_GPU_DECLARE(name)
#else
#define BINARY_GPU_DECLARE(name)\
namespace arraymath_detail {\
template<typename T> void name##_gpu(int N, const T *a, const T *b, T *r);\
template<typename T> void name##_gpu(int N, const T &a, const T *b, T *r);\
template<typename T> void name##_gpu(int N, const T *a, const T &b, T *r);\
}  // namespace arraymath_detail
#endif

// Define a CPU function
#define BINARY_CPU_DEFINE(name) namespace arraymath_detail {\
template<typename T> void name##_cpu(int N, const T *a, const T *b, T *r) {\
  OMP_PARALLEL_FOR\
  for (int i = 0; i < N; i++)\
    r[i] = name(a[i], b[i]);\
}\
template<typename T> void name##_cpu(int N, const T &a, const T *b, T *r) {\
  OMP_PARALLEL_FOR\
  for (int i = 0; i < N; i++)\
    r[i] = name(a, b[i]);\
}\
template<typename T> void name##_cpu(int N, const T *a, const T &b, T *r) {\
  OMP_PARALLEL_FOR\
  for (int i = 0; i < N; i++)\
    r[i] = name(a[i], b);\
}\
}  // namespace arraymath_detail

// Call a GPU function if available
#ifdef CPU_ONLY
#define BINARY_GPU_CALL(name, v)
#else
#define BINARY_GPU_CALL(name, v) if (v.effectiveMode() == AR_GPU)\
  return Expression<T>(makeBinaryAE<T>((F)&name##_gpu<T>, a, b, AR_GPU))
#endif

// Implement unary expressions
#define IMPLEMENT_BINARY(name) BINARY_GPU_DECLARE(name)\
BINARY_CPU_DEFINE(name)\
template<typename T>\
Expression<T> ARMath<T>::name(const ArrayBase<T> & a, const ArrayBase<T> & b) {\
  typedef void (*F)(int, const T*, const T*, T*);\
  using namespace arraymath_detail;/* NOLINT(build/namespaces) */\
  BINARY_GPU_CALL(name, a);\
  return Expression<T>(makeBinaryAE<T>((F)&name##_cpu<T>, a, b, AR_CPU));\
}\
template<typename T>\
Expression<T> ARMath<T>::name(T a, const ArrayBase<T> & b) {\
  typedef void (*F)(int, const T&, const T*, T*);\
  using namespace arraymath_detail;/* NOLINT(build/namespaces) */\
  BINARY_GPU_CALL(name, b);\
  return Expression<T>(makeBinaryAE<T>((F)&name##_cpu<T>, a, b, AR_CPU));\
}\
template<typename T>\
Expression<T> ARMath<T>::name(const ArrayBase<T> & a, T b) {\
  typedef void (*F)(int, const T*, const T &, T*);\
  using namespace arraymath_detail;/* NOLINT(build/namespaces) */\
  BINARY_GPU_CALL(name, a);\
  return Expression<T>(makeBinaryAE<T>((F)&name##_cpu<T>, a, b, AR_CPU));\
}
#define DEFINE_BINARY_OP(name, OP) namespace arraymath_detail {\
  template<typename T> T name(T a, T b) { return a OP b; }\
}  // namespace arraymath_detail
#define IMPLEMENT_BINARY_OP(name, OP) DEFINE_BINARY_OP(name, OP)\
IMPLEMENT_BINARY(name)
#define DEFINE_BINARY_F(name, F) namespace arraymath_detail {\
  template<typename T> T name(T a, T b) { return F(a, b); }\
}  // namespace arraymath_detail
#define IMPLEMENT_BINARY2(name, F) DEFINE_BINARY_F(name, F)\
IMPLEMENT_BINARY(name)

IMPLEMENT_BINARY_OP(add, +);
IMPLEMENT_BINARY_OP(sub, -);
IMPLEMENT_BINARY_OP(mul, *);
IMPLEMENT_BINARY_OP(div, /);
IMPLEMENT_BINARY2(maximum, std::max);
IMPLEMENT_BINARY2(minimum, std::min);
IMPLEMENT_BINARY(pow);

#undef DEFINE_BINARY_OP
#undef IMPLEMENT_BINARY_OP
#undef DEFINE_BINARY_F
#undef IMPLEMENT_BINARY2
#undef IMPLEMENT_BINARY
#undef BINARY_GPU_CALL
#undef BINARY_CPU_DEFINE
#undef BINARY_GPU_DECLARE


/****** Unary array operations ******/
namespace arraymath_detail {
template<typename T>
class PartialReductionAE: public Expression<T>::Implementation {
 public:
  typedef void(*Func)(int, const T *, T *, const ArrayShape &, int);
 protected:
  ArrayBasePointer<T> a_;
  int axis_;
  Func f_;
  static ArrayShape newShape(ArrayShape s, int axis) {
    CHECK_GE(axis, 0) << "Positive axis required";
    CHECK_LT(axis, s.size()) << "Axis out of bounds";
    s[axis] = 1;
    return s;
  }
 public:
  PartialReductionAE(Func f, const ArrayBase<T> &a, int axis, const ArrayMode &m)
    : Expression<T>::Implementation(newShape(a.shape(), axis), m), a_(a),
      axis_(axis), f_(f) {
    CHECK_NE(m, AR_DEFAULT) << "Array mode cannot be default!";
  }
  virtual void evaluate(Array<T> *target) const {
    int N = count(this->shape_);
    Array<T> a = a_.eval();
    f_(N, getData(&a, this->mode()), getMutableData(target, this->mode()),
       a.shape(), axis_);
  }
};
}  // namespace arraymath_detail

// Define a CPU function
#define REDUCTION_CPU_DEFINE(name) namespace arraymath_detail {\
template<typename T> T name##_cpu(int N, const T *a) {\
  T r = a[0];\
  for (int i = 1; i < N; i++)\
    r = name(a[i], r);\
  return r;\
}\
}  // namespace arraymath_detail

// Implement unary expressions
#define IMPLEMENT_REDUCTION(name) REDUCTION_CPU_DEFINE(name)\
template<typename T>\
T ARMath<T>::name(const ArrayBase<T> & ab) {\
  using namespace arraymath_detail; /* NOLINT */\
  Array<T> a = ab.eval();\
  return name##_cpu<T>(count(a.shape()), a.cpu_data());\
}

namespace arraymath_detail {
  template<typename T> T max(const T &a, const T &b) { return a > b ? a : b; }
  template<typename T> T min(const T &a, const T &b) { return a < b ? a : b; }
  template<typename T> T sum(const T &a, const T &b) { return a + b; }
};

IMPLEMENT_REDUCTION(max);
IMPLEMENT_REDUCTION(min);
// IMPLEMENT_REDUCTION(soft_max);
// IMPLEMENT_REDUCTION(soft_min);
IMPLEMENT_REDUCTION(sum);
#undef IMPLEMENT_REDUCTION
#undef REDUCTION_CPU_DEFINE

/**** gemm ****/
namespace arraymath_detail {
void gemm_cpu(bool tA, bool tB, const int M, const int N, const int K,
  float alpha, const float *A, int lda, const float *B, int ldb,
  float beta, float *C, int ldc) {
  cblas_sgemm(CblasRowMajor, tA ? CblasTrans : CblasNoTrans,
    tB ? CblasTrans : CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta,
    C, ldc);
}
void gemm_cpu(bool tA, bool tB, const int M, const int N, const int K,
  double alpha, const double *A, int lda, const double *B, int ldb,
  double beta, double *C, int ldc) {
  cblas_dgemm(CblasRowMajor, tA ? CblasTrans : CblasNoTrans,
    tB ? CblasTrans : CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta,
    C, ldc);
}
#ifndef CPU_ONLY
void gemm_gpu(bool tA, bool tB, const int M, const int N, const int K,
  float alpha, const float *A, int lda, const float *B, int ldb,
  float beta, float *C, int ldc);
void gemm_gpu(bool tA, bool tB, const int M, const int N, const int K,
  double alpha, const double *A, int lda, const double *B, int ldb,
  double beta, double *C, int ldc);
#endif
}  // namespace arraymath_detail

template<typename T>
void ARMath<T>::gemm(bool ta, bool tb, T alpha, const Array<T> &a,
  const Array<T> &b, T beta, Array<T> *c) {
  // Check the dimensions
  CHECK_EQ(a.shape().size(), 2) << "GEMM: 2D array required";
  CHECK_EQ(b.shape().size(), 2) << "GEMM: 2D array required";
  const int M = ta ? a.shape()[1] : a.shape()[0];
  const int K = ta ? a.shape()[0] : a.shape()[1];
  const int N = tb ? b.shape()[0] : b.shape()[1];
  CHECK_EQ(K, tb ? b.shape()[1] : b.shape()[0]) << "GEMM: Dimensions mismatch";
  if (c->shape().size() == 0)
    *c = Array<T>(make_shape(M, N), a.mode());
  CHECK_EQ(c->shape().size(), 2) << "GEMM: 2D array required";
  CHECK_EQ(c->shape()[0], M) << "GEMM: Dimensions mismatch";
  CHECK_EQ(c->shape()[1], N) << "GEMM: Dimensions mismatch";

  // Check and set the mode
  ArrayMode m = c->effectiveMode();
  CHECK_NE(m, AR_DEFAULT) << "GEMM: Array mode cannot be default!";
  if (m != a.effectiveMode() || m != b.effectiveMode())
    LOG(WARNING) << "GEMM: Mixing CPU and GPU mode";

  int lda = ta ? M : K;
  int ldb = tb ? K : N;
#ifndef CPU_ONLY
  if (m == AR_GPU)
    return arraymath_detail::gemm_gpu(ta, tb, M, N, K, alpha, a.gpu_data(),
      lda, b.gpu_data(), ldb, beta, c->mutable_gpu_data(), N);
#endif
  return arraymath_detail::gemm_cpu(ta, tb, M, N, K, alpha, a.cpu_data(),
    lda, b.cpu_data(), ldb, beta, c->mutable_cpu_data(), N);
}

template<typename T>
void ARMath<T>::im2col(const Array<T> &data, int kernel_h, int kernel_w,
  int pad_h, int pad_w, int stride_h, int stride_w, Array<T> *data_col) {
  CHECK_EQ(data.shape().size(), 3) << "im2col: Only 3D arrays supported";
  const int C = data.shape()[0], H = data.shape()[1], W = data.shape()[2];
  const int kernel_dim = kernel_h * kernel_w * C;
  const int out_h = (H + 2 * pad_h - kernel_h) / stride_h + 1;
  const int out_w = (W + 2 * pad_w - kernel_w) / stride_w + 1;
  if (data_col->shape().size() == 0)
    *data_col = Array<T>(make_shape(kernel_dim, out_h, out_w), data.mode());
  CHECK_EQ(data_col->shape()[0], kernel_dim) << "im2col: Kernel dim mismatch";
  CHECK_EQ(data_col->shape()[1], out_h) << "im2col: Height mismatch";
  CHECK_EQ(data_col->shape()[2], out_w) << "im2col: Width mismatch";
  if (data_col->effectiveMode() != data.effectiveMode())
    LOG(WARNING) << "im2col: Mixing CPU and GPU mode!";
#ifndef CPU_ONLY
  if (data.effectiveMode() == AR_GPU)
    return im2col_gpu(data.gpu_data(), C, H, W, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, data_col->mutable_gpu_data());
#endif
  im2col_cpu(data.cpu_data(), C, H, W, kernel_h, kernel_w, pad_h, pad_w,
    stride_h, stride_w, data_col->mutable_cpu_data());
}
template<typename T>
void ARMath<T>::col2im(const Array<T> &data_col, int patch_h, int patch_w,
  int pad_h, int pad_w, int stride_h, int stride_w, Array<T> *data) {
  CHECK_EQ(data->shape().size(), 3) << "col2im: Only 3D arrays supported";
  const int C = data->shape()[0], H = data->shape()[1], W = data->shape()[2];
  const int kernel_dim = patch_h * patch_w * C;
  const int out_h = (H + 2 * pad_h - patch_h) / stride_h + 1;
  const int out_w = (W + 2 * pad_w - patch_w) / stride_w + 1;
  CHECK_EQ(data_col.shape().size(), 3) << "col2im: Only 3D arrays supported";
  CHECK_EQ(data_col.shape()[0], kernel_dim) << "col2im: Kernel dim mismatch";
  CHECK_EQ(data_col.shape()[1], out_h) << "col2im: Height mismatch";
  CHECK_EQ(data_col.shape()[2], out_w) << "col2im: Width mismatch";
  if (data_col.effectiveMode() != data->effectiveMode())
    LOG(WARNING) << "col2im: Mixing CPU and GPU mode!";
#ifndef CPU_ONLY
  if (data_col.effectiveMode() == AR_GPU)
    return col2im_gpu(data_col.gpu_data(), C, H, W, patch_h, patch_w, pad_h,
      pad_w, stride_h, stride_w, data->mutable_gpu_data());
#endif
  col2im_cpu(data_col.cpu_data(), C, H, W, patch_h, patch_w, pad_h, pad_w,
    stride_h, stride_w, data->mutable_cpu_data());
}
template<typename T>
void ARMath<T>::conv(const Array<T> & im, const Array<T> & kernel, int pad_h,
                     int pad_w, int stride_h, int stride_w, Array<T> * out) {
  int D = im.shape().size();
  CHECK_GE(D, 3) << "conv: At least 3 input dimensions required";
  CHECK_EQ(kernel.shape().size(), 4) << "conv: Only 4D kernels supported";
  const int C = im.shape()[D-3], H = im.shape()[D-2], W = im.shape()[D-1];
  const int group = C / kernel.shape()[1];
  CHECK_EQ(group*kernel.shape()[1], C) << "conv: Kernel and im chan mismatch";

  const int kernel_h = kernel.shape()[2], kernel_w = kernel.shape()[3];
  const int out_c = kernel.shape()[0];
  const int out_h = (H + 2 * pad_h - kernel_h) / stride_h + 1;
  const int out_w = (W + 2 * pad_w - kernel_w) / stride_w + 1;
  if (out->shape().size() == 0) {
    ArrayShape s = im.shape();
    s[D-3] = out_c;
    s[D-2] = out_h;
    s[D-1] = out_w;
    *out = Array<T>(s, im.mode());
  }
  CHECK_EQ(D, out->shape().size()) << "conv: Shape dim mismatch";
  for (int d = 0; d < D-3; d++)
    CHECK_EQ(im.shape()[d], out->shape()[d]) << "conv: In/out shape mismatch";
  CHECK_EQ(out->shape()[D-3], out_c) << "conv: In/output shape mismatch";
  CHECK_EQ(out->shape()[D-2], out_h) << "conv: In/output shape mismatch";
  CHECK_EQ(out->shape()[D-1], out_w) << "conv: In/output shape mismatch";
  // Reshape both input and output to a 4d array
  Array<T> im_4d = im.reshape(make_shape(-1, C, H, W));
  Array<T> out_4d = out->reshape(make_shape(-1, out_c, out_h, out_w));
  // Setup the column buffer
  ArrayShape cb_s = make_shape(C*kernel_w*kernel_h, out_h, out_w);
  Array<T> cb(temporaryMemory(count(cb_s)*sizeof(T)), cb_s, im.mode());
  for (int i = 0; i < im_4d.shape()[0]; i++) {
    // Run im2col
    Array<T> im_slice = im_4d[i], out_slice = out_4d[i];
    im2col(im_slice, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, &cb);
    // Compute the GEMM (handles groups implicitly using reshape)
    Array<T> kernel_2d = kernel.reshape(make_shape(out_c, -1));
    Array<T> cb_2d = cb.reshape(make_shape(kernel_2d.shape()[1], -1));
    Array<T> out_2d = out_slice.reshape(make_shape(out_c, -1));
    gemm(false, false, T(1.0), kernel_2d, cb_2d, T(0.0), &out_2d);
  }
}

INSTANTIATE_CLASS(ARMath);

}  // namespace caffe
