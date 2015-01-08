#ifndef CAFFE_SPARSE_BLOB_HPP_
#define CAFFE_SPARSE_BLOB_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

namespace caffe {

template<typename Dtype>
class SparseBlob : public Blob<Dtype> {
 public:
  SparseBlob()
      : Blob<Dtype>(),
        indices_(),
        ptr_(),
        nzz_(0) {
  }

  explicit SparseBlob(const int num, const int channels, const int nzz);

  virtual void Reshape(const int num, const int channels, const int height,
                       const int width);

  void Reshape(const int num, const int channels, const int nzz);

  virtual void ReshapeLike(const Blob<Dtype>& other);

  virtual inline int height() const {
    return 1;
  }
  virtual inline int width() const {
    return 1;
  }
  inline int nzz() const {
    return nzz_;
  }

  virtual inline int offset(const int n, const int c = 0, const int h = 0,
                            const int w = 0) const {
    LOG(FATAL)<< "Offset not supported in sparse blob.";
    return 0;
  }

  virtual inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    LOG(FATAL) << "data_at not implemented yet.";
    return (Dtype)0;
  }

  virtual inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    LOG(FATAL) << "Diff data is not supported in sparse blob.";
    return (Dtype)0;
  }

  inline const shared_ptr<SyncedMemory>& indices() const {
    CHECK(indices_);
    return indices_;
  }

  inline const shared_ptr<SyncedMemory>& ptr() const {
    CHECK(ptr_);
    return ptr_;
  }

  const int* cpu_indices() const;
  const int* cpu_ptr() const;

  const int* gpu_indices() const;
  const int* gpu_ptr() const;

  int* mutable_cpu_indices();
  int* mutable_cpu_ptr();

  int* mutable_gpu_indices();
  int* mutable_gpu_ptr();

  virtual void set_cpu_data(Dtype* data);
  virtual void set_gpu_data(Dtype* data);

  // the num and channels are assumed to be the same but
  // nzz might change that is why is an argument
  // also the actual size of data and indices might exceed nzz
  // to allow for easy slicing.
  // If total_size is -1 is assumed to be equal to nzz
  void set_cpu_data(Dtype* data, int* indices, int* ptr, int nzz,
                     int total_size=-1);
  void set_gpu_data(Dtype* data, int* indices, int* ptr, int nzz,
                     int total_size=-1);

  virtual const Dtype* cpu_diff() const;
  virtual const Dtype* gpu_diff() const;
  virtual Dtype* mutable_cpu_diff();
  virtual Dtype* mutable_gpu_diff();

  virtual void ShareData(const Blob<Dtype>& other);
  virtual void ShareDiff(const Blob<Dtype>& other);
  virtual void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  virtual void Update();
  virtual void FromProto(const BlobProto& proto);
  virtual void ToProto(BlobProto* proto, bool write_diff = false) const;

 protected:
  shared_ptr<SyncedMemory> indices_;
  shared_ptr<SyncedMemory> ptr_;
  int nzz_;

  DISABLE_COPY_AND_ASSIGN(SparseBlob);
};  // class SparseBlob

}  // namespace caffe

#endif  // CAFFE_SPARSE_BLOB_HPP_
