#ifndef CAFFE_ARRAY_MATH_HPP_
#define CAFFE_ARRAY_MATH_HPP_

#include "caffe/array/base.hpp"
#include "caffe/array/math_functions.hpp"

namespace caffe {

#define INSTANTIATE_UNARY_F(N, n) template class Unary<float, N<float> >;\
  template class Unary<double, N<double> >;
#define INSTANTIATE_UNARY LIST_UNARY(INSTANTIATE_UNARY_F)

#define INSTANTIATE_BINARY_F(N, n) template class Binary<float, N<float> >;\
  template class Binary<double, N<double> >;
#define INSTANTIATE_BINARY LIST_BINARY(INSTANTIATE_BINARY_F)

#define INSTANTIATE_REDUCTION_F(N, n)\
  template class Reduction<float, N<float> >;\
  template class Reduction<double, N<double> >;
#define INSTANTIATE_REDUCTION LIST_REDUCTION(INSTANTIATE_REDUCTION_F)

#define INSTANTIATE_ALL INSTANTIATE_UNARY INSTANTIATE_BINARY\
  INSTANTIATE_REDUCTION

template<typename T, typename F>
struct Unary {
  static void eval_cpu(const Array<T> & a, Array<T> * c);
#ifndef CPU_ONLY
  static void eval_gpu(const Array<T> & a, Array<T> * c);
#endif

 public:
  static void eval(const Array<T> & a, Array<T> * c, ArrayMode m = AR_DEFAULT);
};
template<typename T, typename F>
class Binary {
  static void eval_cpu(const Array<T> & a, const Array<T> & b, Array<T> * c);
  static void eval_cpu(T a, const Array<T> & b, Array<T> * c);
  static void eval_cpu(const Array<T> & a, T b, Array<T> * c);
#ifndef CPU_ONLY
  static void eval_gpu(const Array<T> & a, const Array<T> & b, Array<T> * c);
  static void eval_gpu(T a, const Array<T> & b, Array<T> * c);
  static void eval_gpu(const Array<T> & a, T b, Array<T> * c);
#endif

 public:
  static void eval(const Array<T> & a, const Array<T> & b, Array<T> * c,
                   ArrayMode m = AR_DEFAULT);
  static void eval(T a, const Array<T> & b, Array<T> * c,
                   ArrayMode m = AR_DEFAULT);
  static void eval(const Array<T> & a, T b, Array<T> * c,
                   ArrayMode m = AR_DEFAULT);
};
template<typename T, typename F>
class Reduction {
  static T eval_cpu(const Array<T> & a);
#ifndef CPU_ONLY
  static T eval_gpu(const Array<T> & a);
#endif

 public:
  static T eval(const Array<T> & a, ArrayMode m = AR_DEFAULT);
};

}  // namespace caffe

#endif  // CAFFE_ARRAY_MATH_HPP_
