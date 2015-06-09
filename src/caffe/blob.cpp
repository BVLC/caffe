#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/device.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
                          const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

template<typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    count_ *= shape[i];
    shape_[i] = shape[i];
  }
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
}

template<typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

template<typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.shape());
}

template<typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
                  const int width)
    // capacity_ must be initialized before calling Reshape
    : capacity_(0) {
  Reshape(num, channels, height, width);
}

template<typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
    // capacity_ must be initialized before calling Reshape
    : capacity_(0) {
  Reshape(shape);
}

template<typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*) data_->cpu_data();
}

template<typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  data_->set_cpu_data(data);
}

template<typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*) data_->gpu_data();
}

template<typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return (const Dtype*) diff_->cpu_data();
}

template<typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return (const Dtype*) diff_->gpu_data();
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

template<typename Dtype>
const Dtype* Blob<Dtype>::data(Caffe::Brew device) const {
  switch (device) {
  case Caffe::GPU:
    return gpu_data();
  case Caffe::CPU:
  default:
    return cpu_data();
  }
}

template<typename Dtype>
const Dtype* Blob<Dtype>::diff(Caffe::Brew device) const {
  switch (device) {
  case Caffe::GPU:
    return gpu_diff();
  case Caffe::CPU:
  default:
    return cpu_diff();
  }
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_data(Caffe::Brew device) {
  switch (device) {
  case Caffe::GPU:
    return mutable_gpu_data();
  case Caffe::CPU:
  default:
    return mutable_cpu_data();
  }
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_diff(Caffe::Brew device) {
  switch (device) {
  case Caffe::GPU:
    return mutable_gpu_diff();
  case Caffe::CPU:
  default:
    return mutable_cpu_diff();
  }
}

template<typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template<typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template<> void Blob<unsigned int>::Update() {
  NOT_IMPLEMENTED;
}
template<> void Blob<int>::Update() {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  Caffe::Brew brew;
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    brew = Caffe::CPU;
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    brew = Caffe::GPU;
    break;
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
  GetDevice<Dtype>(brew)->axpy(count_, Dtype(-1),
                               static_cast<const Dtype*>(diff_->data(brew)),
                               static_cast<Dtype*>(data_->mutable_data(brew)));
}

template<> unsigned int Blob<unsigned int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template<> int Blob<int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template<typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
  if (!data_) {
    return 0;
  }
  Dtype asum = 0;
  Caffe::Brew brew;
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    brew = Caffe::CPU;
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    brew = Caffe::GPU;
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  GetDevice<Dtype>(brew)->asum(count_, data(brew), &asum);
  return asum;
}

template<> unsigned int Blob<unsigned int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template<> int Blob<int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template<typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
  if (!diff_) {
    return 0;
  }
  Dtype asum = 0;
  Caffe::Brew brew;
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    brew = Caffe::CPU;
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    brew = Caffe::GPU;
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  GetDevice<Dtype>(brew)->asum(count_, diff(brew), &asum);
  return asum;
}

template<> unsigned int Blob<unsigned int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template<> int Blob<int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template<typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
  Dtype sumsq;
  const Dtype* data_ptr;
  if (!data_) {
    return 0;
  }
  Caffe::Brew brew;
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    brew = Caffe::CPU;
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    brew = Caffe::GPU;
    break;
#else
    NO_GPU;
#endif
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  data_ptr = data(brew);
  GetDevice<Dtype>(brew)->dot(count_, data_ptr, data_ptr, &sumsq);
  return sumsq;
}

template<> unsigned int Blob<unsigned int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template<> int Blob<int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template<typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
  Dtype sumsq;
  const Dtype* diff_ptr;
  if (!diff_) {
    return 0;
  }
  Caffe::Brew brew;
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    brew = Caffe::CPU;
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    brew = Caffe::GPU;
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  diff_ptr = diff(brew);
  GetDevice<Dtype>(brew)->dot(count_, diff_ptr, diff_ptr, &sumsq);
  return sumsq;
}

template<> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template<> void Blob<int>::scale_data(int scale_factor) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  Dtype* data;
  if (!data_) {
    return;
  }
  Caffe::Brew brew;
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    brew = Caffe::CPU;
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    brew = Caffe::GPU;
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  data = mutable_data(brew);
  GetDevice<Dtype>(brew)->scal(count_, scale_factor, data);
}

template<> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template<> void Blob<int>::scale_diff(int scale_factor) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  Dtype* diff;
  if (!diff_) {
    return;
  }
  Caffe::Brew brew;
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    brew = Caffe::CPU;
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    brew = Caffe::GPU;
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  diff = mutable_diff(brew);
  GetDevice<Dtype>(brew)->scal(count_, scale_factor, diff);
}

template<typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() || other.has_height()
      || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 && LegacyShape(-4) == other.num()
        && LegacyShape(-3) == other.channels()
        && LegacyShape(-2) == other.height()
        && LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

template<typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  Caffe::Brew brew;
  switch (Caffe::mode()) {
  case Caffe::GPU:
    brew = Caffe::GPU;
    break;
  case Caffe::CPU:
    brew = Caffe::CPU;
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
  if (copy_diff) {
    GetDevice<Dtype>(brew)->copy(
        count_, source.diff(brew),
        static_cast<Dtype*>(diff_->mutable_data(brew)));
  } else {
    GetDevice<Dtype>(brew)->copy(
        count_, source.data(brew),
        static_cast<Dtype*>(data_->mutable_data(brew)));
  }
}

template<typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() || proto.has_height()
        || proto.has_width()) {
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
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  for (int i = 0; i < count_; ++i) {
    data_vec[i] = proto.data(i);
  }
  if (proto.diff_size() > 0) {
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}

template<typename Dtype>
void Blob<Dtype>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const Dtype* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const Dtype* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace caffe

