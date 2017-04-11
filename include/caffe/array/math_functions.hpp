#ifndef CAFFE_ARRAY_MATH_FUNCTIONS_HPP_
#define CAFFE_ARRAY_MATH_FUNCTIONS_HPP_

#include "caffe/common.hpp"

// This file contains the definition of all mathematical functions used.
// To add a new function simply add a struct corresponding to the function
// and list it as either a Unary, Binary or Reduction function. All definitions
// will be updated accordingly.


namespace caffe {

#ifdef __CUDACC__
#define AM_CALL __host__ __device__
#else
#define AM_CALL
#endif

template<typename T>
struct UnaryFunction {
  AM_CALL static T eval(T a) { NOT_IMPLEMENTED; }
};
template<typename T>
struct BinaryFunction {
  AM_CALL static T eval(T a, T b) { NOT_IMPLEMENTED; }
};

////// Unary Functions //////
template<typename T>
struct Abs: public UnaryFunction<T> {
  AM_CALL static T eval(T a) { return a > 0 ? a : -a; }
};
template<typename T>
struct Exp: public UnaryFunction<T> {
  AM_CALL static T eval(T a) { return exp(a); }
};
template<typename T>
struct Log: public UnaryFunction<T> {
  AM_CALL static T eval(T a) { return log(a); }
};
template<typename T>
struct Negate: public UnaryFunction<T> {
  AM_CALL static T eval(T a) { return -a; }
};
template<typename T>
struct Sign: public UnaryFunction<T> {
  AM_CALL static T eval(T a) { return a < 0 ? -1 : a > 0 ? 1 : 0; }
};
template<typename T>
struct Sqrt: public UnaryFunction<T> {
  AM_CALL static T eval(T a) { return sqrt(a); }
};
// To define a new unary function define a struct analogous to the ones above
// and add it to the list below. F is a function that gets called as:
//   F(struct name, common name)
#define LIST_UNARY(F) F(Abs, abs) F(Exp, exp) F(Log, log) F(Negate, negate)\
  F(Sign, sign) F(Sqrt, sqrt)

////// Binary Functions //////
template<typename T>
struct Add: public BinaryFunction<T> {
  AM_CALL static T eval(T a, T b) { return a+b; }
};
template<typename T>
struct Sub: public BinaryFunction<T> {
  AM_CALL static T eval(T a, T b) { return a-b; }
};
template<typename T>
struct Mul: public BinaryFunction<T> {
  AM_CALL static T eval(T a, T b) { return a*b; }
};
template<typename T>
struct Div: public BinaryFunction<T> {
  AM_CALL static T eval(T a, T b) { return a/b; }
};
template<typename T>
struct Max: public BinaryFunction<T> {
  AM_CALL static T eval(T a, T b) { return a > b ? a : b; }
};
template<typename T>
struct Min: public BinaryFunction<T> {
  AM_CALL static T eval(T a, T b) { return a < b ? a : b; }
};
template<typename T>
struct Pow: public BinaryFunction<T> {
  AM_CALL static T eval(T a, T b) { return pow(a, b); }
};
// To define a new binary function define a struct analogous to the ones above
// and add it to the list below. F is a function that gets called as:
//   F(struct name, common name)
#define LIST_BINARY(F) F(Add, add) F(Sub, sub) F(Mul, mul) F(Div, div)\
  F(Max, maximum) F(Min, minimum) F(Pow, pow)

// To define a new reduction function define a binary struct (see above)
// and add it to the list below. F is a function that gets called as:
//   F(struct name, common name)
#define LIST_REDUCTION(F) F(Add, sum) F(Mul, prod) F(Min, min) F(Max, max)

}  // namespace caffe

#endif  // CAFFE_ARRAY_MATH_FUNCTIONS_HPP_
