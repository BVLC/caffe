#ifndef CAFFE_HDF5_DATA_LAYER_HPP_
#define CAFFE_HDF5_DATA_LAYER_HPP_

#include "hdf5.h"

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template<typename Dtype, typename MItype, typename MOtype>
class HDF5DataLayer : public Layer<Dtype, MItype, MOtype> {
 public:
  explicit HDF5DataLayer(const LayerParameter& param)
      : Layer<Dtype, MItype, MOtype>(param), offset_() {}
  virtual ~HDF5DataLayer();
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {}

  virtual inline const char* type() const { return "HDF5Data"; }
  virtual inline int_tp ExactNumBottomBlobs() const { return 0; }
  virtual inline int_tp MinTopBlobs() const { return 1; }

 protected:
  void Next();
  bool Skip();

  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom) {}
  virtual void LoadHDF5FileData(const char* filename);

  vector<string> hdf_filenames_;
  uint_tp num_files_;
  uint_tp current_file_;
  hsize_t current_row_;
  vector<shared_ptr<Blob<Dtype> > > hdf_blobs_;
  vector<uint_tp> data_permutation_;
  vector<uint_tp> file_permutation_;
  uint64_t offset_;
};

}  // namespace caffe

#endif  // CAFFE_HDF5_DATA_LAYER_HPP_
