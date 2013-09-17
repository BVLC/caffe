#ifndef CAFFEINE_BLOB_HPP
#define CAFFEINE_BLOB_HPP

#include "caffeine/common.hpp"
#include "caffeine/syncedmem.hpp"
#include "caffeine/proto/layer_param.pb.h"

namespace caffeine {

template <typename Dtype>
class Blob {
 public:
  Blob()
       : num_(0), channels_(0), height_(0), width_(0), count_(0), data_(),
       diff_() {};
  explicit Blob(const int num, const int height,
    const int width, const int channels);
  virtual ~Blob() {};
  void Reshape(const int num, const int height,
      const int width, const int channels);
  inline int num() { return num_; }
  inline int height() { return height_; }
  inline int width() { return width_; }
  inline int channels() { return channels_; }
  inline int count() {return count_; }
  
  const Dtype* cpu_data();
  const Dtype* gpu_data();
  const Dtype* cpu_diff();
  const Dtype* gpu_diff();
  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();
  void Update();
  void FromProto(const BlobProto& proto);
  void ToProto(BlobProto* proto);
 private:
  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> diff_;
  int num_;
  int height_;
  int width_;
  int channels_;
  int count_;
};  // class Blob

}  // namespace caffeine

#endif  // CAFFEINE_BLOB_HPP_
