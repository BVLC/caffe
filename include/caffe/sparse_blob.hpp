#ifndef CAFFE_SPARSE_BLOB_HPP_
#define CAFFE_SPARSE_BLOB_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

namespace caffe {

/**
 * @brief  An extension of Blob to support sparse data in row CRF format
 *
 * TODO(dox): more thorough description.
 */
template<typename Dtype>
class SparseBlob : public Blob<Dtype> {
 public:
  SparseBlob()
      : Blob<Dtype>(),
        indices_(), ptr_(), nnz_(0) {}

  explicit SparseBlob(const vector<int>& shape, const int nnz);
  explicit SparseBlob(const int num, const int channels, const int nnz);
  
  virtual ~SparseBlob() {}
  
  virtual void Reshape(const int num, const int channels,
                                  const int height, const int width);
  virtual void Reshape(const vector<int>& shape);
  void Reshape(const vector<int>& shape, const int nnz);
  virtual void ReshapeLike(const Blob<Dtype>& other);

  inline int nnz() const {
    return nnz_;
  }

  virtual void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  virtual inline int offset(const vector<int>& indices) const {
    LOG(FATAL)<< "Offset not supported in sparse blob.";
    return 0;
  }
  virtual inline int offset(const int n, const int c = 0, const int h = 0,
        const int w = 0) const {
    LOG(FATAL)<< "Offset not supported in sparse blob.";
    return 0;
  }
  virtual inline Dtype data_at(const int n, const int c, const int h,
        const int w) const {
    LOG(FATAL)<< "data_at not supported in sparse blob.";
    return 0;
  }
  virtual inline Dtype diff_at(const int n, const int c, const int h,
        const int w) const {
    LOG(FATAL)<< "diff_at not supported in sparse blob.";
    return 0;
  }
  virtual inline Dtype data_at(const vector<int>& index) const {
    LOG(FATAL)<< "data_at not supported in sparse blob.";
    return 0;
  }
  virtual inline Dtype diff_at(const vector<int>& index) const {
    LOG(FATAL)<< "diff_at not supported in sparse blob.";
    return 0;
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
  virtual const int* gpu_shape() const;

  // the num and channels are assumed to be the same but
  // nnz might change that is why is an argument
  // also the actual size of data and indices might exceed nnz
  // to allow for easy slicing.
  // If total_size is -1 is assumed to be equal to nnz
  void set_cpu_data(Dtype* data, int* indices, int* ptr, int nnz,
                     int total_size=-1);
  void set_gpu_data(Dtype* data, int* indices, int* ptr, int nnz,
                     int total_size=-1);

  virtual const Dtype* cpu_diff() const;
  virtual const Dtype* gpu_diff() const;
  virtual Dtype* mutable_cpu_diff();
  virtual Dtype* mutable_gpu_diff();

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  virtual Dtype asum_data() const;
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  virtual Dtype asum_diff() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  virtual Dtype sumsq_data() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  virtual Dtype sumsq_diff() const;

  /// @brief Scale the blob data by a constant factor.
  virtual void scale_data(Dtype scale_factor);
  /// @brief Scale the blob diff by a constant factor.
  virtual void scale_diff(Dtype scale_factor);

  virtual void ShareData(const Blob<Dtype>& other);
  virtual void ShareDiff(const Blob<Dtype>& other);

  virtual void Update();
  virtual void FromProto(const BlobProto& proto, bool reshape = true);
  virtual void ToProto(BlobProto* proto, bool write_diff = false) const;

 protected:
  shared_ptr<SyncedMemory> indices_;
  shared_ptr<SyncedMemory> ptr_;
  int nnz_;

  DISABLE_COPY_AND_ASSIGN(SparseBlob);
};  // class SparseBlob

}  // namespace caffe

#endif  // CAFFE_SPARSE_BLOB_HPP_
