#include <boost/make_shared.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <string>
#include <vector>
#include "caffe/array/array.hpp"
#include "caffe/array/math.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename T>
Array<T>::Array(const Array & o) : ArrayMemory(o), ArrayBase<T>(o) { }

template<typename T>
Array<T>::Array(ArrayMode mode) : ArrayMemory(), ArrayBase<T>(mode) { }

template<typename T>
Array<T>::Array(const ArrayShape &shape, ArrayMode mode):
  ArrayMemory(count(shape)*sizeof(T)), ArrayBase<T>(shape, mode) { }

template<typename T>
Array<T>::Array(SyncedMemory *memory, const ArrayShape &shape,
  ArrayMode mode):ArrayMemory(memory, count(shape)), ArrayBase<T>(shape, mode) {
  CHECK_GE(memory->size(), count(shape) *sizeof(T)) << "SyncedMemory size '"
      << memory->size() << "' is smaller than shape " << shapeToString(shape)
      << " with element size " << sizeof(T);
}

template<typename T>
Array<T>::Array(shared_ptr<SyncedMemory> memory, const ArrayShape &shape,
  ArrayMode mode):ArrayMemory(memory, count(shape)), ArrayBase<T>(shape, mode) {
  CHECK_GE(memory->size(), count(shape)*sizeof(T)) << "SyncedMemory size '"
      << memory->size() << "' is smaller than shape " << shapeToString(shape)
      << " with element size " << sizeof(T);
}

template<typename T>
Array<T>::Array(shared_ptr<SyncedMemory> m, size_t o, const ArrayShape &s,
  ArrayMode mode):ArrayMemory(m, o*sizeof(T), count(s)*sizeof(T)),
  ArrayBase<T>(s, mode) {
  CHECK_GE(m->size(), (o+count(s))*sizeof(T)) << "SyncedMemory size '"
      << m->size() << "' is smaller than shape " << shapeToString(s)
      << " with element size " << sizeof(T) << " and offset " << o;
}

template<typename T>
Array<T>::~Array() {}

template<typename T>
void Array<T>::initialize(const ArrayShape &shape) {
  CHECK_EQ(count(this->shape_), 0) << "Array already initialized!";
  this->shape_ = shape;
  ArrayMemory::initializeMemory(count(shape) * sizeof(T));
}

template<typename T>
void Array<T>::setMode(ArrayMode mode) {
  this->mode_ = mode;
}

template <typename T>
void Array<T>::FromProto(const BlobProto& proto, bool reshape) {
  ArrayShape shape;
  if (proto.has_num() || proto.has_channels() ||
      proto.has_height() || proto.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    shape.resize(4);
    shape[0] = proto.num();
    shape[1] = proto.channels();
    shape[2] = proto.height();
    shape[3] = proto.width();
  } else {
    shape.resize(proto.shape().dim_size());
    for (int i = 0; i < proto.shape().dim_size(); ++i) {
      shape[i] = proto.shape().dim(i);
    }
  }
  if (reshape)
    initialize(shape);
  else
    CHECK_EQ(shape, this->shape_) << "shape mismatch (reshape not set)";
  // copy data
  T* data_vec = mutable_cpu_data();
  for (int i = 0; i < count(this->shape_); i++)
    data_vec[i] = proto.data(i);
  CHECK_EQ(proto.diff_size(), 0) << "Cannot read BlobProto diff";
}

template <typename T>
void Array<T>::ToProto(BlobProto* proto) const {
  proto->clear_shape();
  for (int i = 0; i < this->shape_.size(); i++) {
    proto->mutable_shape()->add_dim(this->shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const T* data_vec = cpu_data();
  for (int i = 0; i < count(this->shape_); i++)
    proto->add_data(data_vec[i]);
}
template<typename T>
Array<T> Array<T>::eval() const {
  return *this;
}
template<typename T>
shared_ptr<SyncedMemory> Array<T>::memory() const {
  return memory_;
}
template<typename T>
const T *Array<T>::cpu_data() const {
  return static_cast<const T *>(ArrayMemory::cpu_data_());
}
template<typename T>
const T *Array<T>::gpu_data() const {
  return static_cast<const T *>(ArrayMemory::gpu_data_());
}
template<typename T>
T *Array<T>::mutable_cpu_data() {
  return static_cast<T *>(ArrayMemory::mutable_cpu_data_());
}
template<typename T>
T *Array<T>::mutable_gpu_data() {
  return static_cast<T *>(ArrayMemory::mutable_gpu_data_());
}
template<typename T>
Array<T> &Array<T>::operator=(const Expression<T> & other) {
  if (!memory_) {
    initialize(other.shape());
    setMode(other.mode());
  }
  CHECK_EQ(this->shape(), other.shape()) << "Array shape missmatches";
  other.evaluate(this);
  return *this;
}
template<typename T>
Array<T> &Array<T>::operator=(const T &v) {
  CHECK(memory_) << "Array not initialized";
#ifndef CPU_ONLY
  if (this->effectiveMode() == AR_GPU)
    caffe_gpu_set(count(this->shape()), v, this->mutable_gpu_data());
  else
#endif
    caffe_set(count(this->shape()), v, this->mutable_cpu_data());
  return *this;
}
template<typename T>
Array<T> &Array<T>::operator=(const Array<T> &other) {
  if (!memory_) {
    initialize(other.shape());
    setMode(other.mode());
  }
  CHECK_EQ(this->shape(), other.shape()) << "Array shape missmatches";
#ifndef CPU_ONLY
  if (this->effectiveMode() == AR_GPU)
    // NOLINT_NEXT_LINE(caffe/alt_fn)
    CUDA_CHECK(cudaMemcpy(this->mutable_gpu_data(), other.gpu_data(),
                          sizeof(T) * count(this->shape()), cudaMemcpyDefault));
  else
#endif
    // NOLINT_NEXT_LINE(caffe/alt_fn)
    memcpy(this->mutable_cpu_data(), other.cpu_data(),
           sizeof(T) * count(this->shape()));
  return *this;
}
template<typename T>
Array<T> Array<T>::reshape(ArrayShape shape) const {
  size_t p = 1;
  int md = -1;
  for (int d = 0; d < shape.size(); d++)
    if (shape[d] == -1) {
      CHECK_EQ(md, -1) << "Only one missing dimension supported";
      md = d;
    } else {
      p *= shape[d];
    }
  if (md >= 0) shape[md] = count(this->shape()) / p;
  CHECK_EQ(count(this->shape()), count(shape)) <<
    "reshape cannot change array size";
  return Array<T>(memory_, offset_/sizeof(T), shape, this->mode());
}
template<typename T>
Array<T> Array<T>::operator[](size_t d) {
  CHECK_GT(this->shape().size(), 0) << "At least one dimension required";
  CHECK_LT(d, this->shape()[0]) << "Index out of range";
  ArrayShape s(this->shape().begin()+1, this->shape().end());
  return Array<T>(memory_, d*count(s), s, this->mode());
}
template<typename T>
const Array<T> Array<T>::operator[](size_t d) const {
  CHECK_GT(this->shape().size(), 0) << "At least one dimension required";
  CHECK_LT(d, this->shape()[0]) << "Index out of range";
  ArrayShape s(this->shape().begin()+1, this->shape().end());
  return Array<T>(memory_, d*count(s), s, this->mode());
}

INSTANTIATE_CLASS(Array);
}  // namespace caffe
