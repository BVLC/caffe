#include "caffe/array/base.hpp"
#include <boost/make_shared.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <string>
#include <vector>
#include "caffe/array/array.hpp"
#include "caffe/array/expression.hpp"

namespace std {
ostream &operator<<(ostream &os, const caffe::ArrayShape &shape) {
  return os << caffe::shapeToString(shape);
}
}  // namespace std

namespace caffe {

#define MAX_TEMPORARY_MEMORY 1000
static boost::mutex temporary_memory_mtx;
struct tmp_delete {
  bool * in_use;
  explicit tmp_delete(bool * in_use):in_use(in_use) {}
  void operator()(void const *) const {
    // Allow the memory to be reused
    boost::lock_guard<boost::mutex> guard(temporary_memory_mtx);
    *in_use = false;
  }
};
shared_ptr< SyncedMemory > temporaryMemory(size_t size) {
  // Avoid race conditions
  boost::lock_guard<boost::mutex> guard(temporary_memory_mtx);
  static shared_ptr<SyncedMemory> memory[MAX_TEMPORARY_MEMORY];
  static bool in_use[MAX_TEMPORARY_MEMORY] = {0};
  for (int i = 0; i < MAX_TEMPORARY_MEMORY; i++)
    if (!in_use[i]) {
      in_use[i] = true;
      if (!memory[i] || memory[i]->size() < size)
        memory[i] = boost::make_shared<SyncedMemory>(size);
      return shared_ptr<SyncedMemory>(memory[i].get(), tmp_delete(&in_use[i]));
    }
  LOG(FATAL) << "All temporary memory used! Consider increasing "
    "MAX_TEMPORARY_MEMORY.";
  return shared_ptr< SyncedMemory >();
}

struct no_delete {
  void operator()(void const *) const {}
};
ArrayMemory::ArrayMemory():offset_(0), size_(0) {
}
ArrayMemory::ArrayMemory(size_t size):offset_(0), size_(0) {
  initializeMemory(size);
}
ArrayMemory::ArrayMemory(SyncedMemory *memory, size_t size): offset_(0),
  size_(size >0 ? size : memory->size()), memory_(memory, no_delete()) {
}
ArrayMemory::ArrayMemory(shared_ptr<SyncedMemory> memory, size_t size):
  offset_(0), size_(size >0 ? size : memory->size()), memory_(memory) {
}
ArrayMemory::ArrayMemory(shared_ptr<SyncedMemory> memory, size_t offset,
  size_t size): offset_(offset), size_(size >0 ? size : memory->size()),
  memory_(memory) {
}
ArrayMemory::~ArrayMemory() {
}
const void *ArrayMemory::cpu_data_() const {
  return !memory_ ? NULL : static_cast<const char*>(memory_->cpu_data()) +
    offset_;
}
const void *ArrayMemory::gpu_data_() const {
  return !memory_ ? NULL : static_cast<const char*>(memory_->gpu_data()) +
    offset_;
}
void *ArrayMemory::mutable_cpu_data_() {
  return !memory_ ? NULL : static_cast<char*>(memory_->mutable_cpu_data()) +
    offset_;
}
void *ArrayMemory::mutable_gpu_data_() {
  return !memory_ ? NULL : static_cast<char*>(memory_->mutable_gpu_data()) +
    offset_;
}
size_t ArrayMemory::size() {
  return size_;
}
bool ArrayMemory::isBorrowed() const {
  return boost::get_deleter<no_delete>(memory_);
}
void ArrayMemory::initializeMemory(size_t size) {
  CHECK_EQ(isBorrowed(), false) << "Cannot initialize borrowed memory!";
  CHECK(!memory_) << "Memory already initialized!";
  size_ = size;
  memory_ = boost::make_shared<SyncedMemory>(size);
}

///// Operation Mode /////
ArrayMode globalArrayMode() {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU)
    return AR_GPU;
#endif
  return AR_CPU;
}

///// Array Shape /////
ArrayShape make_shape(size_t d0) {
  ArrayShape r;
  r.push_back(d0);
  return r;
}
ArrayShape make_shape(size_t d0, size_t d1) {
  ArrayShape r;
  r.push_back(d0);
  r.push_back(d1);
  return r;
}
ArrayShape make_shape(size_t d0, size_t d1, size_t d2) {
  ArrayShape r;
  r.push_back(d0);
  r.push_back(d1);
  r.push_back(d2);
  return r;
}
ArrayShape make_shape(size_t d0, size_t d1, size_t d2, size_t d3) {
  ArrayShape r;
  r.push_back(d0);
  r.push_back(d1);
  r.push_back(d2);
  r.push_back(d3);
  return r;
}
ArrayShape make_shape(size_t d0, size_t d1, size_t d2, size_t d3, size_t d4) {
  ArrayShape r;
  r.push_back(d0);
  r.push_back(d1);
  r.push_back(d2);
  r.push_back(d3);
  r.push_back(d4);
  return r;
}
ArrayShape make_shape(size_t d0, size_t d1, size_t d2, size_t d3, size_t d4,
                      size_t d5) {
  ArrayShape r;
  r.push_back(d0);
  r.push_back(d1);
  r.push_back(d2);
  r.push_back(d3);
  r.push_back(d4);
  r.push_back(d5);
  return r;
}
size_t count(const ArrayShape &shape) {
  if (shape.size() == 0) return 0;
  size_t r = 1;
  for (size_t i = 0; i < shape.size(); i++)
    r *= shape[i];
  return r;
}
string shapeToString(const ArrayShape &shape) {
  ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); i++)
    oss << shape[i] << (i + 1 < shape.size() ? ", " : "]");
  return oss.str();
}

template<typename T> ArrayBase<T>::Reference::Reference(ArrayShape shape,
  ArrayMode mode):shape(shape), mode(mode) {}
template<typename T> ArrayBase<T>::Reference::~Reference() {}
template<typename T>
ArrayBase<T>::ArrayBase(ArrayMode mode):mode_(mode) {}

template<typename T>
ArrayBase<T>::ArrayBase(const ArrayShape &shape, ArrayMode mode)
  :shape_(shape), mode_(mode) {}

template<typename T>
ArrayBase<T>::~ArrayBase() {
}

template<typename T>
const ArrayShape & ArrayBase<T>::shape() const {
  return shape_;
}

template<typename T>
ArrayMode ArrayBase<T>::mode() const {
  return mode_;
}

template<typename T>
ArrayMode ArrayBase<T>::effectiveMode() const {
  if (mode_ == AR_DEFAULT)
    return globalArrayMode();
  return mode_;
}
// Define all unary functions
#define DEFINE_UNARY(F, f)\
template<typename T> Expression<T> ArrayBase<T>::f() const {\
  return Expression<T>::template Unary<F<T> >(this->ref());\
}
  LIST_UNARY(DEFINE_UNARY);
#undef DEFINE_UNARY

// Define all binary functions
#define DEFINE_BINARY(F, f)\
template<typename T> Expression<T> ArrayBase<T>::f(const ArrayBase &o) const {\
  return Expression<T>::template Binary<F<T> >(this->ref(), o.ref());\
}\
template<typename T> Expression<T> ArrayBase<T>::f(T o) const {\
  return Expression<T>::template Binary<F<T> >(this->ref(), o);\
}
  LIST_BINARY(DEFINE_BINARY);
#undef DEFINE_BINARY
template<typename T> Expression<T> ArrayBase<T>::isub(T o) const {
  return Expression<T>::template Binary<Sub<T> >(o, this->ref());
}
template<typename T> Expression<T> ArrayBase<T>::idiv(T o) const {
  return Expression<T>::template Binary<Div<T> >(o, this->ref());
}

// Define all reductions
#define DEFINE_REDUCTION(F, f)\
template<typename T> T ArrayBase<T>::f() const {\
  return Reduction<T, F<T> >::eval(eval());\
}
  LIST_REDUCTION(DEFINE_REDUCTION);
#undef DEFINE_REDUCTION
template<typename T> T ArrayBase<T>::mean() const {
  return sum() / count(this->shape());
}

INSTANTIATE_CLASS(ArrayBase);
}  // namespace caffe
