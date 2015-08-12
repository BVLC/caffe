#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Blob<Dtype>::Init() {
  data_.reset(new Tensor<Dtype>());
  diff_.reset(new Tensor<Dtype>());
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  data_->Reshape(shape);
  diff_->Reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape) {
  ASSERT(shape.dim_size() <= kMaxBlobAxes, "");
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.shape());
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width) {
  Init();
  Reshape(num, channels, height, width);
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape) {
  Init();
  Reshape(shape);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  ASSERT(data_, "");
  return (const Dtype*)data_->cpu_mem();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  ASSERT(data, "");
  data_->set_cpu_mem(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  ASSERT(data_, "");
  return (const Dtype*)data_->gpu_mem();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  ASSERT(diff_, "");
  return (const Dtype*)diff_->cpu_mem();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  ASSERT(diff_, "");
  return (const Dtype*)diff_->gpu_mem();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  ASSERT(data_, "");
  return static_cast<Dtype*>(data_->mutable_cpu_mem());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  ASSERT(data_, "");
  return static_cast<Dtype*>(data_->mutable_gpu_mem());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  ASSERT(diff_, "");
  return static_cast<Dtype*>(diff_->mutable_cpu_mem());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  ASSERT(diff_, "");
  return static_cast<Dtype*>(diff_->mutable_gpu_mem());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  ASSERT(count() == other.count(), "");
  data_->ShareMem(*other.data());
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
  ASSERT(count() == other.count(), "");
  diff_->ShareMem(*other.diff());
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.

template <typename Dtype>
void Blob<Dtype>::Update() {
  Update(Dtype(1));
}
template <typename Dtype>
void Blob<Dtype>::Update(Dtype lr) {
  data_->AddMulFromDynamicMode(*diff_, Dtype(-lr));
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
  if (!data_) { return 0; }
  return data_->asum();
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
  if (!diff_) { return 0; }
  return diff_->asum();
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
  if (!data_) { return 0; }
  return data_->sumsq();
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
  if (!diff_) { return 0; }
  return diff_->sumsq();
}

template <typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  if (!data_) { return; }
  data_->scale(scale_factor);
}

template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  if (!diff_) { return; }
  diff_->scale(scale_factor);
}

template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape().size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape() == other_shape;
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count() || source.shape() != shape()) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      ASSERT(false, "Trying to copy blobs of different sizes.");
    }
  }
  if (copy_diff) {
    diff_->CopyFrom(*source.diff());
  } else {
    data_->CopyFrom(*source.data());
  }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
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
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  if (proto.double_data_size() > 0) {
    CHECK_EQ(count(), proto.double_data_size());
    for (int i = 0; i < count(); ++i) {
      data_vec[i] = proto.double_data(i);
    }
  } else {
    CHECK_EQ(count(), proto.data_size());
    for (int i = 0; i < count(); ++i) {
      data_vec[i] = proto.data(i);
    }
  }
  if (proto.double_diff_size() > 0) {
    CHECK_EQ(count(), proto.double_diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count(); ++i) {
      diff_vec[i] = proto.double_diff(i);
    }
  } else if (proto.diff_size() > 0) {
    CHECK_EQ(count(), proto.diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count(); ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}

template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape().size(); ++i) {
    proto->mutable_shape()->add_dim(shape()[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  const double* data_vec = cpu_data();
  for (int i = 0; i < count(); ++i) {
    proto->add_double_data(data_vec[i]);
  }
  if (write_diff) {
    const double* diff_vec = cpu_diff();
    for (int i = 0; i < count(); ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape().size(); ++i) {
    proto->mutable_shape()->add_dim(shape()[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const float* data_vec = cpu_data();
  for (int i = 0; i < count(); ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const float* diff_vec = cpu_diff();
    for (int i = 0; i < count(); ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

template <typename Dtype>
void Blob<Dtype>::SetDataValues(const Dtype value) {
  data_->SetValues(value);
}

template <typename Dtype>
void Blob<Dtype>::SetDiffValues(const Dtype value) {
  diff_->SetValues(value);
}

template <typename Dtype>
void Blob<Dtype>::AddDataFrom(const Blob& source) {
  data_->AddFrom(*source.data());
}

template <typename Dtype>
void Blob<Dtype>::AddDiffFrom(const Blob& source) {
  diff_->AddFrom(*source.diff());
}

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace caffe
