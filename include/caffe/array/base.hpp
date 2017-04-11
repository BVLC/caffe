#ifndef CAFFE_ARRAY_BASE_HPP_
#define CAFFE_ARRAY_BASE_HPP_

#include <string>
#include <vector>
#include "caffe/array/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

namespace std {
// Print function for ArrayShape
ostream &operator<<(ostream &os, const vector<int> &shape);
}
namespace caffe {

// Return a temporary SyncedMemory, useful to evaluate nested expressions
shared_ptr<SyncedMemory> temporaryMemory(size_t size);

class ArrayMemory {
 protected:
  size_t offset_, size_;
  shared_ptr<SyncedMemory> memory_;
  void initializeMemory(size_t size);

 public:
  ArrayMemory();
  explicit ArrayMemory(size_t size);
  explicit ArrayMemory(shared_ptr<SyncedMemory> memory, size_t size = 0);
  ArrayMemory(shared_ptr<SyncedMemory> memory, size_t offset, size_t size);
  // Avoid using this constructor, as it leads to serious issues if
  // ArrayMemory outlives memory!
  explicit ArrayMemory(SyncedMemory *memory, size_t size = 0);
  virtual ~ArrayMemory();
  virtual const void *cpu_data_() const;
  virtual const void *gpu_data_() const;
  virtual void *mutable_cpu_data_();
  virtual void *mutable_gpu_data_();
  bool isBorrowed() const;
  virtual size_t size();
};

// Shape functions
typedef std::vector<int> ArrayShape;
ArrayShape make_shape(size_t d0);
ArrayShape make_shape(size_t d0, size_t d1);
ArrayShape make_shape(size_t d0, size_t d1, size_t d2);
ArrayShape make_shape(size_t d0, size_t d1, size_t d2, size_t d3);
ArrayShape make_shape(size_t d0, size_t d1, size_t d2, size_t d3, size_t d4);
ArrayShape make_shape(size_t d0, size_t d1, size_t d2, size_t d3, size_t d4,
                      size_t d5);
size_t count(const ArrayShape &shape);
std::string shapeToString(const ArrayShape &shape);

enum ArrayMode {
  AR_DEFAULT = 0,
  AR_CPU = 1,
  AR_GPU = 2
};
ArrayMode globalArrayMode();

template<typename Dtype> class Expression;
template<typename Dtype> class Array;

template<typename Dtype>
class ArrayBase {
 protected:
  // The ArrayBase reference allows any function to reference a ArrayBase object
  // without having to copy it and without worrying that it might get freed.
  struct Reference {
    ArrayShape shape;
    ArrayMode mode;
    Reference(ArrayShape shape, ArrayMode mode);
    virtual ~Reference();
    virtual Array<Dtype> eval() const = 0;
  };
  virtual shared_ptr<Reference> ref() const = 0;
  ArrayShape shape_;
  ArrayMode mode_;

 public:
  explicit ArrayBase(ArrayMode mode = AR_DEFAULT);
  explicit ArrayBase(const ArrayShape &shape, ArrayMode mode = AR_DEFAULT);
  virtual ~ArrayBase();

  virtual Array<Dtype> eval() const = 0;

  virtual const ArrayShape & shape() const;
  virtual ArrayMode mode() const;
  // Get the Array mode GPU or CPU, no default mode allowed
  virtual ArrayMode effectiveMode() const;

  // Define all unary functions
#define DECLARE_UNARY(F, f)  Expression<Dtype> f() const;
  LIST_UNARY(DECLARE_UNARY)
#undef DECLARE_UNARY

  // Define all binary functions
#define DECLARE_BINARY(F, f)\
  Expression<Dtype> f(const ArrayBase &other) const;\
  Expression<Dtype> f(Dtype other) const;
  LIST_BINARY(DECLARE_BINARY)
#undef DECLARE_BINARY
  Expression<Dtype> isub(Dtype o) const;
  Expression<Dtype> idiv(Dtype o) const;

  // Define various operators
  Expression<Dtype> operator+(const ArrayBase &o) const { return add(o); }
  Expression<Dtype> operator+(Dtype o) const { return add(o); }
  Expression<Dtype> operator-(const ArrayBase &o) const { return sub(o); }
  Expression<Dtype> operator-(Dtype o) const { return sub(o); }
  Expression<Dtype> operator*(const ArrayBase &o) const { return mul(o); }
  Expression<Dtype> operator*(Dtype o) const { return mul(o); }
  Expression<Dtype> operator/(const ArrayBase &o) const { return div(o); }
  Expression<Dtype> operator/(Dtype o) const { return div(o); }
  Expression<Dtype> operator-() const { return negate(); }

  // Define all reductions
#define DECLARE_REDUCTION(F, f)  Dtype f() const;
  LIST_REDUCTION(DECLARE_REDUCTION)
#undef DECLARE_REDUCTION
  Dtype mean() const;
};
template<typename T> Expression<T> operator+(T a, const ArrayBase<T> &b) {
  return b.add(a);
}
template<typename T> Expression<T> operator-(T a, const ArrayBase<T> &b) {
  return b.isub(a);
}
template<typename T> Expression<T> operator*(T a, const ArrayBase<T> &b) {
  return b.mul(a);
}
template<typename T> Expression<T> operator/(T a, const ArrayBase<T> &b) {
  return b.idiv(a);
}

}  // namespace caffe

#endif  // CAFFE_ARRAY_BASE_HPP_
