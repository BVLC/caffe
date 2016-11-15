#ifndef CAFFE_MEMORY_SPARSE_DATA_LAYER_HPP_
#define CAFFE_MEMORY_SPARSE_DATA_LAYER_HPP_

#include <vector>

#include "caffe/sparse_blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from memory.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MemorySparseDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit MemorySparseDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param), has_new_data_(false) {}
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MemorySparseData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

  virtual void AddDatumVector(const vector<SparseDatum>& datum_vector);

  // Reset should accept const pointers, but can't, because the memory
  //  will be given to Blob, which is mutable
  void Reset(Dtype* data, int *indices, int *ptr, Dtype* label, int n, int nnz);
  void set_batch_size(int new_size);

  int batch_size() { return batch_size_; }
  int channels() { return channels_; }
  int height() { return height_; }
  int width() { return width_; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  int batch_size_, channels_, height_, width_, size_;
  Dtype* data_;
  int *indices_;
  int *ptr_;
  Dtype* labels_;
  int nnz_;
  int n_;
  size_t pos_;
  size_t data_pos_;
  size_t indices_pos_;
  size_t ptr_pos_;
  size_t nnz_pos_;
  SparseBlob<Dtype> added_data_;
  Blob<Dtype> added_label_;
  bool has_new_data_;
};

}  // namespace caffe

#endif  // CAFFE_MEMORY_SPARSE_DATA_LAYER_HPP_
