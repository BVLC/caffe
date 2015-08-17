#include <boost/make_shared.hpp>
#include "caffe/array/array.hpp"
#include "caffe/array/math.hpp"

using boost::make_shared;
namespace caffe {

template<typename T, typename F>
void Unary<T, F>::eval_cpu(const Array<T> & a, Array<T> * c) {
  CHECK(!!c) << "Output array does not exist!";
  CHECK_EQ(a.shape(), c->shape()) << "Shape does not match!";
  const int N = count(a.shape());
  const T * pa = a.cpu_data();
  T * pc = c->mutable_cpu_data();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N; i++)
    pc[i] = F::eval(pa[i]);
}
template<typename T, typename F>
void Unary<T, F>::eval(const Array<T> & a, Array<T> * c, ArrayMode m) {
  if (m == AR_DEFAULT) m = a.effectiveMode();
#ifndef CPU_ONLY
  if (m == AR_GPU)
    return eval_gpu(a, c);
#endif
  return eval_cpu(a, c);
}

template<typename T, typename F>
void Binary<T, F>::eval_cpu(const Array<T> &a, const Array<T> &b, Array<T> *c) {
  CHECK(!!c) << "Output array does not exist!";
  CHECK_EQ(a.shape(), c->shape()) << "Shape does not match!";
  CHECK_EQ(b.shape(), c->shape()) << "Shape does not match!";
  const int N = count(a.shape());
  const T * pa = a.cpu_data(), * pb = b.cpu_data();
  T * pc = c->mutable_cpu_data();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N; i++)
    pc[i] = F::eval(pa[i], pb[i]);
}
template<typename T, typename F>
void Binary<T, F>::eval_cpu(T a, const Array<T> & b, Array<T> * c) {
  CHECK(!!c) << "Output array does not exist!";
  CHECK_EQ(b.shape(), c->shape()) << "Shape does not match!";
  const int N = count(b.shape());
  const T * pb = b.cpu_data();
  T * pc = c->mutable_cpu_data();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N; i++)
    pc[i] = F::eval(a, pb[i]);
}
template<typename T, typename F>
void Binary<T, F>::eval_cpu(const Array<T> & a, T b, Array<T> * c) {
  CHECK(!!c) << "Output array does not exist!";
  CHECK_EQ(a.shape(), c->shape()) << "Shape does not match!";
  const int N = count(a.shape());
  const T * pa = a.cpu_data();
  T * pc = c->mutable_cpu_data();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N; i++)
    pc[i] = F::eval(pa[i], b);
}
template<typename T, typename F>
void Binary<T, F>::eval(const Array<T> & a, const Array<T> & b, Array<T> * c,
                        ArrayMode m) {
  if (m == AR_DEFAULT) m = a.effectiveMode();
  if (m != b.effectiveMode()) LOG(WARNING) << "Mixing CPU and GPU mode";
#ifndef CPU_ONLY
  if (m == AR_GPU)
    return eval_gpu(a, b, c);
#endif
  return eval_cpu(a, b, c);
}
template<typename T, typename F>
void Binary<T, F>::eval(T a, const Array<T> & b, Array<T> * c, ArrayMode m) {
  if (m == AR_DEFAULT) m = b.effectiveMode();
#ifndef CPU_ONLY
  if (m == AR_GPU)
    return eval_gpu(a, b, c);
#endif
  return eval_cpu(a, b, c);
}
template<typename T, typename F>
void Binary<T, F>::eval(const Array<T> & a, T b, Array<T> * c, ArrayMode m) {
  if (m == AR_DEFAULT) m = a.effectiveMode();
#ifndef CPU_ONLY
  if (m == AR_GPU)
    return eval_gpu(a, b, c);
#endif
  return eval_cpu(a, b, c);
}

template<typename T, typename F>
T Reduction<T, F>::eval_cpu(const Array<T> & a) {
  const int N = count(a.shape());
  CHECK_GT(N, 0) << "At least one element required for reduction";
  const T * pa = a.cpu_data();
  T r = pa[0];
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 1; i < N; i++)
    r = F::eval(pa[i], r);
  return r;
}
template<typename T, typename F>
T Reduction<T, F>::eval(const Array<T> & a, ArrayMode m) {
  if (m == AR_DEFAULT) m = a.effectiveMode();
#ifndef CPU_ONLY
  if (m == AR_GPU)
    return eval_gpu(a);
#endif
  return eval_cpu(a);
}

INSTANTIATE_ALL;

}  // namespace caffe
