#ifndef CAFFE_ARRAY_HPP_
#define CAFFE_ARRAY_HPP_

#include <string>
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "syncedmem.hpp"

namespace std {
// Print function for ArrayShape
ostream &operator<<(ostream &os, const vector<size_t> &shape);
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
  ~ArrayMemory();
  virtual const void *cpu_data_() const;
  virtual const void *gpu_data_() const;
  virtual void *mutable_cpu_data_();
  virtual void *mutable_gpu_data_();
  bool isBorrowed() const;
  virtual size_t size();
};

// Shape functions
typedef std::vector<size_t> ArrayShape;
ArrayShape make_shape(const std::vector<int> & s);
ArrayShape make_shape(size_t d0);
ArrayShape make_shape(size_t d0, size_t d1);
ArrayShape make_shape(size_t d0, size_t d1, size_t d2);
ArrayShape make_shape(size_t d0, size_t d1, size_t d2, size_t d3);
ArrayShape make_shape(size_t d0, size_t d1, size_t d2, size_t d3, size_t d4);
ArrayShape make_shape(size_t d0, size_t d1, size_t d2, size_t d3, size_t d4,
                      size_t d5);
size_t count(const ArrayShape &shape);
std::string shapeToString(const ArrayShape &shape);
bool operator==(const ArrayShape &a, const ArrayShape &b);

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

  // Define various mathematical operations
  Expression<Dtype> operator+(const ArrayBase &other) const;
  Expression<Dtype> operator+(const Dtype &other) const;
  Expression<Dtype> operator-(const ArrayBase &other) const;
  Expression<Dtype> operator-(const Dtype &other) const;
  Expression<Dtype> operator*(const ArrayBase &other) const;
  Expression<Dtype> operator*(const Dtype &other) const;
  Expression<Dtype> operator/(const ArrayBase &other) const;
  Expression<Dtype> operator/(const Dtype &other) const;
  Expression<Dtype> pow(const ArrayBase &other) const;
  Expression<Dtype> pow(const Dtype &other) const;

  Expression<Dtype> abs() const;
  Expression<Dtype> exp() const;
  Expression<Dtype> log() const;
  Expression<Dtype> negate() const;
  Expression<Dtype> operator-() const;
  Expression<Dtype> sign() const;
  Expression<Dtype> sqrt() const;

  // TODO: Implement reductions
};
template<typename Dtype>
class Expression: public ArrayBase<Dtype> {
 public:
  struct Implementation: public ArrayBase<Dtype> {
    Implementation(const ArrayShape &shape, ArrayMode mode = AR_DEFAULT);
    virtual void evaluate(Array<Dtype> *target) const = 0;
    Array<Dtype> eval() const;
  };

 protected:
  shared_ptr<Implementation> imp_;

 public:
  explicit Expression(shared_ptr<Implementation> imp);
  shared_ptr<Implementation> imp() const;
  virtual void evaluate(Array<Dtype> *target) const;
  Array<Dtype> eval() const;
};

template<typename Dtype>
class Array: protected ArrayMemory, public ArrayBase<Dtype> {
 protected:
  void initialize(const ArrayShape &shape);
  Array(shared_ptr<SyncedMemory> memory, size_t offset, const ArrayShape &shape,
    ArrayMode mode);

 public:
  explicit Array(ArrayMode mode = AR_DEFAULT);
  explicit Array(const ArrayShape &shape, ArrayMode mode = AR_DEFAULT);
  Array(shared_ptr<SyncedMemory> memory, const ArrayShape &shape,
    ArrayMode mode = AR_DEFAULT);
  // Avoid using this constructor if Array could ever outlive memory
  Array(SyncedMemory *memory, const ArrayShape &shape,
        ArrayMode mode = AR_DEFAULT);

  Array<Dtype> eval() const;
  void setMode(ArrayMode mode);

  void FromProto(const BlobProto& proto, bool reshape = false);
  void ToProto(BlobProto* proto) const;

  shared_ptr<SyncedMemory> memory() const;
  virtual const Dtype *cpu_data() const;
  virtual const Dtype *gpu_data() const;
  virtual Dtype *mutable_cpu_data();
  virtual Dtype *mutable_gpu_data();

  // Math expression evaluation
  Array &operator=(const Expression<Dtype> &other);
  // Copy/duplicate the data
  Array &operator=(const Array &other);
  // Set the array to a specific value
  Array &operator=(const Dtype &v);
  // Change the shape of the array and share the data
  Array reshape(ArrayShape shape) const;
  // Slicing operator (note: this will reference the original array)
  Array operator[](size_t d);
  const Array operator[](size_t d) const;

  // Define various mathematical operations
  Array &operator+=(const ArrayBase<Dtype> &other);
  Array &operator+=(const Dtype &other);
  Array &operator-=(const ArrayBase<Dtype> &other);
  Array &operator-=(const Dtype &other);
  Array &operator*=(const ArrayBase<Dtype> &other);
  Array &operator*=(const Dtype &other);
  Array &operator/=(const ArrayBase<Dtype> &other);
  Array &operator/=(const Dtype &other);
  Array &powInPlace(const ArrayBase<Dtype> &other);
  Array &powInPlace(const Dtype &other);

  // Define a few unary functions such as log, logInPlace, ...
  Array &absInPlace();
  Array &expInPlace();
  Array &logInPlace();
  Array &negateInPlace();
  Array &signInPlace();
  Array &sqrtInPlace();
};
template<typename T> Expression<T> operator+(const T &a, const ArrayBase<T> &b);
template<typename T> Expression<T> operator-(const T &a, const ArrayBase<T> &b);
template<typename T> Expression<T> operator*(const T &a, const ArrayBase<T> &b);
template<typename T> Expression<T> operator/(const T &a, const ArrayBase<T> &b);

}  // namespace caffe

#endif  // CAFFE_ARRAY_HPP_
