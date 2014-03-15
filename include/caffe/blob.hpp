// Copyright 2013 Yangqing Jia

#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Blob {
 public:
  Blob()
       : data_(), diff_(), num_(0), channels_(0), height_(0), width_(0),
         count_(0), capacity_(0) {}
  explicit Blob(const int num, const int channels, const int height,
    const int width);
  virtual ~Blob() {}
  void Reshape(const int num, const int channels, const int height,
               const int width);
  // Only reshape the num while keeping the other three dims intact
  void ReshapeNum(const int num);
  // Re-allocate memory if the current blob capacity is not big enough
  void Reserve(const size_t capacity);
  inline int num() const { return num_; }
  inline int channels() const { return channels_; }
  inline int height() const { return height_; }
  inline int width() const { return width_; }
  inline int count() const { return count_; }
  inline size_t capacity() const { return capacity_; }
  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {
    return ((n * channels_ + c) * height_ + h) * width_ + w;
  }
  // Copy from source. If copy_diff is false, we copy the data; if copy_diff
  // is true, we copy the diff.
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return *(cpu_data() + offset(n, c, h, w));
  }

  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return *(cpu_diff() + offset(n, c, h, w));
  }

  const Dtype* cpu_data() const;
  const Dtype* gpu_data() const;
  const Dtype* cpu_diff() const;
  const Dtype* gpu_diff() const;
  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();
  inline size_t data_size() const { return data_->size(); }
  inline size_t diff_size() const { return diff_->size(); }
  inline bool has_data() const {
    if (data_) {
      return true;
    }
    return false;
  }
  inline bool has_diff() const {
    if (diff_) {
      return true;
    }
    return false;
  }
  void Update();
  void FromProto(const BlobProto& proto);
  void ToProto(BlobProto* proto, bool write_diff = false) const;

 protected:
  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> diff_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int count_;
  size_t capacity_;

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
