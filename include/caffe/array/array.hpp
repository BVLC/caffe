#ifndef CAFFE_ARRAY_ARRAY_HPP_
#define CAFFE_ARRAY_ARRAY_HPP_

#include <string>
#include <vector>
#include "caffe/array/base.hpp"
#include "caffe/array/expression.hpp"
#include "caffe/array/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
class Array: protected ArrayMemory, public ArrayBase<Dtype> {
 protected:
  struct Reference: public ArrayBase<Dtype>::Reference {
    Reference(shared_ptr<SyncedMemory> memory, size_t offset, ArrayShape shape,
              ArrayMode mode):ArrayBase<Dtype>::Reference(shape, mode),
              memory(memory), offset(offset) {
    }
    shared_ptr<SyncedMemory> memory;
    size_t offset;
    virtual Array<Dtype> eval() const {
      return Array<Dtype>(memory, offset, this->shape, this->mode);
    }
  };
  virtual shared_ptr<typename ArrayBase<Dtype>::Reference> ref() const {
    return shared_ptr<Reference>(new Reference(memory_, offset_, this->shape_,
                                               this->mode_));
  }
  void initialize(const ArrayShape &shape);
  Array(shared_ptr<SyncedMemory> memory, size_t offset, const ArrayShape &shape,
    ArrayMode mode);

 public:
  Array(const Array & o);
  explicit Array(ArrayMode mode = AR_DEFAULT);
  explicit Array(const ArrayShape &shape, ArrayMode mode = AR_DEFAULT);
  Array(shared_ptr<SyncedMemory> memory, const ArrayShape &shape,
    ArrayMode mode = AR_DEFAULT);
  // Avoid using this constructor if Array could ever outlive memory
  Array(SyncedMemory *memory, const ArrayShape &shape,
        ArrayMode mode = AR_DEFAULT);
  virtual ~Array();

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


  // Define a in place unaries
#define DEFINE_UNARY_IN_PLACE(F, f) Array<Dtype>& f##InPlace() {\
    return *this = this->f();\
  }
LIST_UNARY(DEFINE_UNARY_IN_PLACE);
#undef DEFINE_UNARY_IN_PLACE

  // Define a in place binaries
#define DEFINE_BINARY_IN_PLACE(F, f) Array<Dtype>& f##InPlace(Dtype o) {\
    return *this = this->f(o);\
  }\
  Array<Dtype>& f##InPlace(const ArrayBase<Dtype> &o) {\
    return *this = this->f(o);\
  }
LIST_BINARY(DEFINE_BINARY_IN_PLACE);
#undef DEFINE_BINARY_IN_PLACE

  // Define in place operators
  Array &operator+=(const ArrayBase<Dtype> &o) { return *this = *this + o; }
  Array &operator+=(Dtype o) { return *this = *this + o; }
  Array &operator-=(const ArrayBase<Dtype> &o) { return *this = *this - o; }
  Array &operator-=(Dtype o) { return *this = *this - o; }
  Array &operator*=(const ArrayBase<Dtype> &o) { return *this = *this * o; }
  Array &operator*=(Dtype o) { return *this = *this * o; }
  Array &operator/=(const ArrayBase<Dtype> &o) { return *this = *this / o; }
  Array &operator/=(Dtype o) { return *this = *this / o; }
};

}  // namespace caffe

#endif  // CAFFE_ARRAY_ARRAY_HPP_
