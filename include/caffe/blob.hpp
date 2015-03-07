#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * Blob%s are 4-dimensional arrays that store data (e.g. images, model
 * parameters and derivatives) in the order of (Num, Channels, Height, Width).
 * These data are stored in a C-contiguous fashion in CPU and/or GPU memory.
 + Blob%s conceal the computational overhead of mixed CPU/GPU operation.
 * Syncronization between the CPU and the GPU is provided by the SyncedMemory
 * objects, when it is necessary in order to provide communication between
 * layers (e.g. when the network is trained in GPU but one layer does not
 * implement the Layer<Dtype>::Forward_gpu() or the Layer<Dtype>::Backward_gpu() 
 * function).
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Blob {
 public:
  Blob()
       : data_(), diff_(), num_(0), channels_(0), height_(0), width_(0),
       count_(0), capacity_(0) {}
  explicit Blob(const int num, const int channels, const int height,
    const int width);
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
   * @param num
   *     (in general) the number of 3-dimensional arrays or otherwise the batch
   *     size
   * @param channels
   *     (in general) the number of dimensions per 2-dimensional array or
   *     otherwise the number of channels per image
   * @param height
   *     the first dimension of the 2-dimensional array or otherwise the height
   *     per image
   * @param width
   *     the second dimension of the 2-dimensional array or otherwise the width
   *     per image
   */
  void Reshape(const int num, const int channels, const int height,
    const int width);
  /**
   * @brief Change the dimensions of the blob, based on the dimensions of
   *        another blob, while allocating new memory if necessary.
   *
   * @param Blob
   *     the blob which's dimensions are used as a reference to reshape the
   *     callig blob
   */
  void ReshapeLike(const Blob& other);
  inline int num() const { return num_; }
  inline int channels() const { return channels_; }
  inline int height() const { return height_; }
  inline int width() const { return width_; }
  inline int count() const { return count_; }
  /**
   * @brief Return the position of a 4-dimensional array as it is store in a
   *        C-contiguous arrays
   */
  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num_);
    CHECK_GE(channels_, 0);
    CHECK_LE(c, channels_);
    CHECK_GE(height_, 0);
    CHECK_LE(h, height_);
    CHECK_GE(width_, 0);
    CHECK_LE(w, width_);
    return ((n * channels_ + c) * height_ + h) * width_ + w;
  }
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

  /**
   * @brief Return the Dtype value of the data which are in the (n, c, h, w)
   *        position of the blob (the 4-dimensional array)
   */
  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return *(cpu_data() + offset(n, c, h, w));
  }

  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return *(cpu_diff() + offset(n, c, h, w));
  }

  /**
   * @brief return the SyncedMemory pointer to data
   */
  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }

  const Dtype* cpu_data() const;
  void set_cpu_data(Dtype* data);
  const Dtype* gpu_data() const;
  const Dtype* cpu_diff() const;
  const Dtype* gpu_diff() const;
  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();
  void Update();
  void FromProto(const BlobProto& proto);
  void ToProto(BlobProto* proto, bool write_diff = false) const;

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  Dtype asum_data() const;
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  Dtype asum_diff() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  Dtype sumsq_data() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  Dtype sumsq_diff() const;

  /// @brief Scale the blob data by a constant factor.
  void scale_data(Dtype scale_factor);
  /// @brief Scale the blob diff by a constant factor.
  void scale_diff(Dtype scale_factor);

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other);
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareDiff(const Blob& other);

 protected:
  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> diff_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int count_;
  int capacity_;

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
