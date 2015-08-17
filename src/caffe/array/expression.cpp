#include <boost/make_shared.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <string>
#include <vector>
#include "caffe/array/array.hpp"
#include "caffe/array/expression.hpp"
#include "caffe/array/math.hpp"

namespace caffe {

template<typename T>
Expression<T>::Implementation::Implementation(ArrayShape s, ArrayMode m):
ArrayBase<T>::Reference(s, m) { }

template<typename T>
Array<T> Expression<T>::Implementation::eval() const {
  Array<T> r(temporaryMemory(count(this->shape)*sizeof(T)),
                  this->shape, this->mode);
  evaluate(&r);
  return r;
}

template<typename T> Expression<T>::Expression(const ArrayShape &s, ArrayMode m)
  : ArrayBase<T>(s, m) { }
template<typename T>
Array<T> Expression<T>::eval() const {
  Array<T> r(imp_->shape, imp_->mode);
  imp_->evaluate(&r);
  return r;
}
template<typename T>
void Expression<T>::evaluate(Array<T> * r) const {
  imp_->evaluate(r);
}

INSTANTIATE_CLASS(Expression);
}  // namespace caffe
