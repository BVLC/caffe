#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/tensor.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
Blob<Dtype>::Blob()
    : data_tensor_(new SyncedTensor<Dtype>()),
      diff_tensor_(new SyncedTensor<Dtype>()),
      count_(0),
      // capacity_ must be initialized before calling Reshape
      capacity_(0) {
}

template<typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width)
    : data_tensor_(new SyncedTensor<Dtype>()),
      diff_tensor_(new SyncedTensor<Dtype>()),
      // capacity_ must be initialized before calling Reshape
      capacity_(0) {
  Reshape(num, channels, height, width);
}

template<typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
    : data_tensor_(new SyncedTensor<Dtype>()),
      diff_tensor_(new SyncedTensor<Dtype>()),
      // capacity_ must be initialized before calling Reshape
      capacity_(0) {
  Reshape(shape);
}

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
  data_tensor_->Reshape(shape);
  diff_tensor_->Reshape(shape);
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
const Dtype* Blob<Dtype>::cpu_data() const {
  return data_tensor_->cpu_data();
}

template<typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  data_tensor_->set_cpu_data(data);
}

template<typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  return data_tensor_->gpu_data();
}

template<typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  return diff_tensor_->cpu_data();
}

template<typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  return diff_tensor_->gpu_data();
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  return data_tensor_->mutable_cpu_data();
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  return data_tensor_->mutable_gpu_data();
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  return diff_tensor_->mutable_cpu_data();
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  return diff_tensor_->mutable_gpu_data();
}

template<typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  CHECK_EQ(count_, other.count());
  data_tensor_ = other.data_tensor();
}

template<typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
  CHECK_EQ(count_, other.count());
  diff_tensor_ = other.diff_tensor();
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
  data_tensor_->add(Dtype(-1), diff());
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
  return data_tensor_->asum();
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
  return diff_tensor_->asum();
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
  return data_tensor_->sumsq();
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
  return diff_tensor_->sumsq();
}

template<> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template<> void Blob<int>::scale_data(int scale_factor) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  data_tensor_->scale(scale_factor);
}

template<> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template<> void Blob<int>::scale_diff(int scale_factor) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  diff_tensor_->scale(scale_factor);
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
  if (copy_diff) {
    diff_tensor_->CopyFrom(source.diff(), reshape);
  } else {
    data_tensor_->CopyFrom(source.data(), reshape);
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

