#include "caffe/array.hpp"
#include <boost/make_shared.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <string>
#include <vector>
#include "caffe/arraymath.hpp"

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
ArrayShape make_shape(const std::vector<int> & s) {
  std::vector<size_t> r(s.size());
  for (int i = 0; i < s.size(); i++)
    r[i] = s[i];
  return r;
}
ArrayShape make_shape(int d0) {
  std::vector<size_t> r;
  r.push_back(d0);
  return r;
}
ArrayShape make_shape(size_t d0, size_t d1) {
  std::vector<size_t> r;
  r.push_back(d0);
  r.push_back(d1);
  return r;
}
ArrayShape make_shape(size_t d0, size_t d1, size_t d2) {
  std::vector<size_t> r;
  r.push_back(d0);
  r.push_back(d1);
  r.push_back(d2);
  return r;
}
ArrayShape make_shape(size_t d0, size_t d1, size_t d2, size_t d3) {
  std::vector<size_t> r;
  r.push_back(d0);
  r.push_back(d1);
  r.push_back(d2);
  r.push_back(d3);
  return r;
}
ArrayShape make_shape(size_t d0, size_t d1, size_t d2, size_t d3, size_t d4) {
  std::vector<size_t> r;
  r.push_back(d0);
  r.push_back(d1);
  r.push_back(d2);
  r.push_back(d3);
  r.push_back(d4);
  return r;
}
ArrayShape make_shape(size_t d0, size_t d1, size_t d2, size_t d3, size_t d4,
                      size_t d5) {
  std::vector<size_t> r;
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
bool operator==(const std::vector<size_t> &a, const std::vector<size_t> &b) {
  if (a.size() != b.size()) return false;
  for (int i = 0; i < a.size(); i++)
    if (a[i] != b[i])
      return false;
  return true;
}

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

///// Math expressions /////
template<typename T>
Expression<T>::Expression(shared_ptr<Implementation> imp)
  : ArrayBase<T>(imp->shape(), imp->mode()), imp_(imp) {}
template<typename T>
Array<T> Expression<T>::eval() const {
  Array<T> r(temporaryMemory(count(this->shape())*sizeof(T)), this->shape(),
             this->mode());
  imp_->evaluate(&r);
  return r;
}
template<typename T>
void Expression<T>::evaluate(Array<T> * r) const {
  imp_->evaluate(r);
}

template<typename T>
Expression<T>::Implementation::Implementation(const ArrayShape &shape,
  ArrayMode mode):ArrayBase<T>(shape, mode) {}

template<typename T>
Array<T> Expression<T>::Implementation::eval() const {
  Array<T> r(temporaryMemory(count(this->shape())*sizeof(T)), this->shape(),
             this->mode());
  evaluate(&r);
  return r;
}
template<typename T>
shared_ptr<typename Expression<T>::Implementation> Expression<T>::imp() const {
  return imp_;
}


///// Array /////
template<typename T>
Array<T>::Array(ArrayMode mode) : ArrayMemory(), ArrayBase<T>(mode)  {
}
template<typename T>
Array<T>::Array(const ArrayShape &shape, ArrayMode mode):
  ArrayMemory(count(shape)*sizeof(T)), ArrayBase<T>(shape, mode) {
}
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
  if (reshape) {
    initialize(shape);
  } else {
    CHECK_EQ(shape, this->shape_) << "shape mismatch (reshape not set)";
  }
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


// Define the unary operators
#define DEFINE_UNARY(N) \
  template<typename T>\
  Expression<T> ArrayBase<T>::N() const {\
    return ARMath<T>::N(*this);\
  }\
  template<typename T>\
  Array<T>& Array<T>::N##InPlace() {\
    return *this = ARMath<T>::N(*this);\
  }

DEFINE_UNARY(abs);
DEFINE_UNARY(exp);
DEFINE_UNARY(log);
DEFINE_UNARY(negate);
DEFINE_UNARY(sign);
DEFINE_UNARY(sqrt);

template<typename T> Expression<T> ArrayBase<T>::operator-() const {
  return ARMath<T>::negate(*this);
}

#undef DEFINE_UNARY

// Define the binary operators
#define DEFINE_BINARY(N) \
  template<typename T>\
  Expression<T>\
    ArrayBase<T>::N(const ArrayBase<T> & o) const {\
    return ARMath<T>::N(*this, o);\
  }\
  template<typename T>\
  Array<T>& Array<T>::N##InPlace(const ArrayBase<T> & o) {\
    return *this = ARMath<T>::N(*this, o);\
  }\
  template<typename T>\
  Expression<T>\
    ArrayBase<T>::N(const T & o) const {\
    return ARMath<T>::N(*this, o);\
  }\
  template<typename T>\
  Array<T>& Array<T>::N##InPlace(const T & o) {\
    return *this = ARMath<T>::N(*this, o);\
  }

#define DEFINE_BINARY2(N, O, OO) \
  template<typename T>\
  Expression<T>\
    ArrayBase<T>::operator O(const ArrayBase<T> & o) const {\
    return ARMath<T>::N(*this, o);\
  }\
  template<typename T>\
  Array<T>& Array<T>::operator OO(const ArrayBase<T> & o) {\
    return *this = ARMath<T>::N(*this, o);\
  }\
  template<typename T>\
  Expression<T>\
    ArrayBase<T>::operator O(const T & o) const {\
    return ARMath<T>::N(*this, o);\
  }\
  template<typename T>\
  Array<T>& Array<T>::operator OO(const T & o) {\
    return *this = ARMath<T>::N(*this, o);\
  }\
  template<typename T>\
  Expression<T>\
    operator O(const T & a, const ArrayBase<T> &b) {\
    return ARMath<T>::N(a, b);\
  }\
  template Expression<float> operator O(const float & a, \
    const ArrayBase<float> &b);\
  template Expression<double> operator O(const double & a, \
    const ArrayBase<double> &b);

DEFINE_BINARY2(add, +, +=);
DEFINE_BINARY2(sub, -, -=);
DEFINE_BINARY2(mul, *, *=);
DEFINE_BINARY2(div, / , /=);
DEFINE_BINARY(pow);

#undef DEFINE_UNARY


INSTANTIATE_CLASS(ArrayBase);
INSTANTIATE_CLASS(Expression);
INSTANTIATE_CLASS(Array);
}  // namespace caffe
