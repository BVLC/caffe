#ifndef CAFFE_HDF5_OUTPUT_LAYER_HPP_
#define CAFFE_HDF5_OUTPUT_LAYER_HPP_

#include "hdf5.h"

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

#define HDF5_DATA_DATASET_NAME "data"
#define HDF5_DATA_LABEL_NAME "label"

/**
 * @brief Write blobs to disk as HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template<typename Dtype, typename MItype, typename MOtype>
class HDF5OutputLayer : public Layer<Dtype, MItype, MOtype> {
 public:
  explicit HDF5OutputLayer(const LayerParameter& param)
      : Layer<Dtype, MItype, MOtype>(param), file_opened_(false) {}
  virtual ~HDF5OutputLayer();
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {}

  virtual inline const char* type() const { return "HDF5Output"; }
  // TODO: no limit on the number of blobs
  virtual inline int_tp ExactNumBottomBlobs() const { return 2; }
  virtual inline int_tp ExactNumTopBlobs() const { return 0; }

  inline string file_name() const { return file_name_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);
  virtual void SaveBlobs();

  bool file_opened_;
  string file_name_;
  hid_t file_id_;
  Blob<MItype> data_blob_;
  Blob<MOtype> label_blob_;
};

}  // namespace caffe

#endif  // CAFFE_HDF5_OUTPUT_LAYER_HPP_
