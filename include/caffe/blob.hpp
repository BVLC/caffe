#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/backend/vptr.hpp"

const int_tp kMaxBlobAxes = 32;

namespace caffe {

class Device;

class QuantizerBase;

class BlobBase {
 public:
  BlobBase()
      : data_(),
        diff_(),
        count_(0),
        capacity_(0),
        device_(Caffe::GetDefaultDevice()) {
  }
  BlobBase(Device *dev)
      : data_(),
        diff_(),
        count_(0),
        capacity_(0),
        device_(dev) {
  }
  virtual ~BlobBase() {

  }


  virtual void FromProto(const BlobProto& proto, bool reshape = true) = 0;
  virtual void ToProto(BlobProto* proto, bool write_diff = false) const = 0;

  virtual void Update() = 0;
  virtual void Clear() = 0;

  virtual bool Reshape(const vector<int_tp>& shape) = 0;
  virtual bool Reshape(const vector<int_tp>& shape,
               const vector<int_tp>& shape_stride) = 0;
  virtual bool Reshape(const BlobShape& shape) = 0;
  virtual bool Reshape(const BlobShape& shape,
                       const BlobShape& shape_stride) = 0;
  virtual bool Reshape(const int_tp num, const int_tp channels,
                       const int_tp height, const int_tp width) = 0;
  virtual bool ReshapeLike(const BlobBase* other) = 0;

  inline string shape_string() const {
    std::ostringstream stream;
    for (int_tp i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }
  inline const vector<int_tp>& shape() const {
    return shape_;
  }
  inline const vector<int_tp>& shape_stride() const {
    return shape_stride_;
  }
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  inline int_tp shape(int_tp index) const {
    return shape_[CanonicalAxisIndex(index)];
  }
  inline int_tp shape_stride(int_tp index) const {
    return shape_stride_[CanonicalAxisIndex(index)];
  }
  inline int_tp num_axes() const {
    return shape_.size();
  }
  inline int_tp count() const {
    return count_;
  }
  virtual uint_tp byte_count() const = 0;

  inline shared_ptr<QuantizerBase> quant() {
    CHECK(quant_.get());
    return quant_;
  }

  inline void set_quant(shared_ptr<QuantizerBase> quant) {
    CHECK(quant.get());
    quant_ = quant;
  }

  virtual void scale_data(const void* scale_factor) = 0;
  virtual void scale_diff(const void* scale_factor) = 0;

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  inline int_tp count(int_tp start_axis, int_tp end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int_tp count = 1;
    for (int_tp i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }

  inline int_tp count_stride(int_tp start_axis, int_tp end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int_tp count = 1;
    for (int_tp i = start_axis; i < end_axis; ++i) {
      count *= shape_stride(i);
    }
    return count;
  }

  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  inline int_tp count(int_tp start_axis) const {
    return count(start_axis, num_axes());
  }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  inline int_tp CanonicalAxisIndex(int_tp axis_index) const {
    CHECK_GE(axis_index, -num_axes())
        <<"axis " << axis_index
        << " out of range for " << num_axes()
    << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes())
    << "axis " << axis_index << " out of range for " << num_axes()
    << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }

  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  inline int_tp num() const {return LegacyShape(0);}
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  inline int_tp channels() const {return LegacyShape(1);}
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  inline int_tp height() const {return LegacyShape(2);}
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  inline int_tp width() const {return LegacyShape(3);}
  inline int_tp LegacyShape(int_tp index) const {
    CHECK_LE(num_axes(), 4)
    << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }

  inline int_tp offset(const int_tp n, const int_tp c = 0,
                       const int_tp h = 0, const int_tp w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }

  inline int_tp offset(const vector<int_tp>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int_tp offset = 0;
    for (int_tp i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }

  /**
   * @brief Return the device context to which this blob and shared memory belongs
   */
  inline Device *get_device() const {
    return device_;
  }

  bool ShapeEquals(const BlobProto& other);

  virtual DataType data_type() const = 0;

  virtual void asum_data(void* out) const = 0;
  virtual void asum_diff(void* out) const = 0;
  virtual void sumsq_data(void* out) const = 0;
  virtual void sumsq_diff(void* out) const = 0;

  virtual void cpu_data(void* out) const = 0;
  virtual void cpu_diff(void* out) const = 0;
  virtual void gpu_data(vptr<void> out) const = 0;
  virtual void gpu_diff(vptr<void> out) const = 0;

  virtual void set_cpu_data(const void* const in) = 0;
  virtual void set_cpu_diff(const void* const in) = 0;
  virtual void set_gpu_data(vptr<const void> in) = 0;
  virtual void set_gpu_diff(vptr<const void> in) = 0;

  void ShareDataBase(const BlobBase* other);
  void ShareDiffBase(const BlobBase* other);

  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }

 protected:
  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> diff_;
  shared_ptr<SyncedMemory> shape_data_;
  shared_ptr<SyncedMemory> shape_stride_data_;
  vector<int_tp> shape_;
  vector<int_tp> shape_stride_;
  vector<int_tp> offset_shape_;
  int_tp count_;
  int_tp capacity_;
  Device *device_;
  shared_ptr<QuantizerBase> quant_;

  DISABLE_COPY_AND_ASSIGN(BlobBase);
};

/**
 * @brief a wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
template<typename Dtype>
class Blob : public BlobBase {
 public:
  Blob() : BlobBase() { }
  virtual ~Blob() { }

  explicit Blob(Device *dev);

  explicit Blob(const int_tp num, const int_tp channels, const int_tp height,
                const int_tp width, Device *device_context =
                    Caffe::GetDefaultDevice());
  explicit Blob(const vector<int_tp>& shape,
                Device *device_context =
                    Caffe::GetDefaultDevice());
  explicit Blob(const vector<int_tp>& shape,
                const vector<int_tp>& shape_stride,
                Device *device_context =
                    Caffe::GetDefaultDevice());

  void Init();

  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   *
   * Reshape returns true if new memory was allocated.
   */
  virtual bool Reshape(const vector<int_tp>& shape);
  virtual bool Reshape(const vector<int_tp>& shape,
               const vector<int_tp>& shape_stride);
  virtual bool Reshape(const BlobShape& shape);
  virtual bool Reshape(const BlobShape& shape,
                       const BlobShape& shape_stride);
  virtual bool Reshape(const int_tp num, const int_tp channels,
                       const int_tp height, const int_tp width);
  virtual bool ReshapeLike(const BlobBase* other);
  bool ReshapeLike(const Blob& other);

  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  inline Dtype data_at(const int_tp n, const int_tp c, const int_tp h,
      const int_tp w) const {
    return cpu_data()[offset(n, c, h, w)];
  }

  inline Dtype diff_at(const int_tp n, const int_tp c, const int_tp h,
      const int_tp w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }

  inline Dtype data_at(const vector<int_tp>& index) const {
    return cpu_data()[offset(index)];
  }

  inline Dtype diff_at(const vector<int_tp>& index) const {
    return cpu_diff()[offset(index)];
  }

  const Dtype* cpu_data() const;
  void set_cpu_data(Dtype* data);
  vptr<const int_tp> gpu_shape() const;
  vptr<const Dtype> gpu_data() const;
  void set_gpu_data(vptr<Dtype> data);
  const Dtype* cpu_diff() const;
  vptr<const Dtype> gpu_diff() const;
  Dtype* mutable_cpu_data();
  vptr<Dtype> mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  vptr<Dtype> mutable_gpu_diff();
  virtual void Update();
  virtual void Clear();

  virtual void FromProto(const BlobProto& proto, bool reshape = true);
  virtual void ToProto(BlobProto* proto, bool write_diff = false) const;

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  Dtype asum_data() const;
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  Dtype asum_diff() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  Dtype sumsq_data() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  Dtype sumsq_diff() const;

  virtual void asum_data(void* out) const;
  virtual void asum_diff(void* out) const;
  virtual void sumsq_data(void* out) const;
  virtual void sumsq_diff(void* out) const;

  virtual void cpu_data(void* out) const;
  virtual void cpu_diff(void* out) const;
  virtual void gpu_data(vptr<void> out) const;
  virtual void gpu_diff(vptr<void> out) const;

  virtual void set_cpu_data(const void* const in);
  virtual void set_cpu_diff(const void* const in);
  virtual void set_gpu_data(vptr<const void> in);
  virtual void set_gpu_diff(vptr<const void> in);

  /// @brief Scale the blob data by a constant factor.
  virtual void scale_data(const void* scale_factor);
  void scale_data(Dtype scale_factor);
  /// @brief Scale the blob diff by a constant factor.
  virtual void scale_diff(const void* scale_factor);
  void scale_diff(Dtype scale_factor);

  virtual uint_tp byte_count() const;

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer&s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other);
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layers which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareDiff(const Blob& other);

  virtual DataType data_type() const;

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob


template<> void Blob<int8_t>::Init();
template<> void Blob<int16_t>::Init();
template<> void Blob<int32_t>::Init();
template<> void Blob<int64_t>::Init();

template<> void Blob<int8_t>::Update();
template<> void Blob<int16_t>::Update();
template<> void Blob<int32_t>::Update();
template<> void Blob<int64_t>::Update();

template<> int8_t Blob<int8_t>::asum_data() const;
template<> int16_t Blob<int16_t>::asum_data() const;
template<> int32_t Blob<int32_t>::asum_data() const;
template<> int64_t Blob<int64_t>::asum_data() const;

template<> int8_t Blob<int8_t>::asum_diff() const;
template<> int16_t Blob<int16_t>::asum_diff() const;
template<> int32_t Blob<int32_t>::asum_diff() const;
template<> int64_t Blob<int64_t>::asum_diff() const;

template<> int8_t Blob<int8_t>::sumsq_data() const;
template<> int16_t Blob<int16_t>::sumsq_data() const;
template<> int32_t Blob<int32_t>::sumsq_data() const;
template<> int64_t Blob<int64_t>::sumsq_data() const;

template<> int8_t Blob<int8_t>::sumsq_diff() const;
template<> int16_t Blob<int16_t>::sumsq_diff() const;
template<> int32_t Blob<int32_t>::sumsq_diff() const;
template<> int64_t Blob<int64_t>::sumsq_diff() const;

template<> void Blob<int8_t>::scale_data(int8_t scale_factor);
template<> void Blob<int16_t>::scale_data(int16_t scale_factor);
template<> void Blob<int32_t>::scale_data(int32_t scale_factor);
template<> void Blob<int64_t>::scale_data(int64_t scale_factor);

template<> void Blob<int8_t>::scale_diff(int8_t scale_factor);
template<> void Blob<int16_t>::scale_diff(int16_t scale_factor);
template<> void Blob<int32_t>::scale_diff(int32_t scale_factor);
template<> void Blob<int64_t>::scale_diff(int64_t scale_factor);

template<> void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const;
template<> void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const;

EXTERN_CLASS_1T(Blob, (half_fp)(float)(double));
EXTERN_CLASS_1T(Blob, (int8_t)(int16_t)(int32_t)(int64_t));
EXTERN_CLASS_1T(Blob, (uint8_t)(uint16_t)(uint32_t)(uint64_t));

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
