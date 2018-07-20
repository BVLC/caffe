#ifndef CAFFE_HDF5_DATA_LAYER_HPP_
#define CAFFE_HDF5_DATA_LAYER_HPP_

#include "hdf5.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

namespace hdf5DataLayerDetail {

  template <typename Dtype>
  class HDF5FileDataBuffer {
    unsigned int file_idx_;
    std::vector<shared_ptr<Blob<Dtype> > > hdf_blobs_;
    std::vector<unsigned int> data_permutation_;

   public:
      HDF5FileDataBuffer(unsigned int idx, const std::string& file,
        LayerParameter* layer_param);

    unsigned int file_idx() const {return file_idx_;}
    const std::vector<shared_ptr<Blob<Dtype> > >& hdf_blobs() const {
     return hdf_blobs_;
    }
    const std::vector<unsigned int>& data_permutation() const {
     return data_permutation_;
    }
  };

  template <typename Dtype>
  class HDF5FileDataHandler {
    int current_file_ = 0;
    std::vector<std::string> hdf_filenames_;
    std::vector<unsigned int> file_permutation_;
    std::weak_ptr<HDF5FileDataBuffer<Dtype>> current_buffer_;

    std::mutex loadDataMutex_;

    LayerParameter* layer_param_;

   public:
      HDF5FileDataHandler(const std::vector<std::string>& files,
        LayerParameter* layer_param);

    const std::vector<std::string>& files() const {return hdf_filenames_;}

    /*!
     * @param prev last buffer the caller went through
     * */
    std::shared_ptr<HDF5FileDataBuffer<Dtype>> getBuffer(
      std::shared_ptr<HDF5FileDataBuffer<Dtype>> prev);
  };

  template <typename Dtype>
  class HDF5DataManager {
    static void createInstance();
    static std::unique_ptr<HDF5DataManager<Dtype>> _instance;
    static std::mutex _createInstanceMutex;

   public:
    inline static HDF5DataManager& instance() {
      if (!_instance)
        createInstance();
      return *_instance;
    }

    HDF5FileDataHandler<Dtype>* registerFileSet(
      const std::vector<std::string>& files, LayerParameter* layer_param);

   protected:
    std::unordered_set<std::unique_ptr<HDF5FileDataHandler<Dtype>>> handlers_;
    std::mutex instanceMutex_;
  };

  template<class Dtype>
  std::unique_ptr<HDF5DataManager<Dtype>> HDF5DataManager<Dtype>::_instance;
  template<class Dtype>
  std::mutex HDF5DataManager<Dtype>::_createInstanceMutex;
}  // namespace hdf5DataLayerDetail

/**
 * @brief Provides data to the Net from HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class HDF5DataLayer : public Layer<Dtype> {
 public:
  explicit HDF5DataLayer(const LayerParameter& param)
      : Layer<Dtype>(param), offset_() {}
  virtual ~HDF5DataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "HDF5Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  void Next();
  bool Skip();

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  hdf5DataLayerDetail::HDF5FileDataHandler<Dtype>* data_handler_;
  std::shared_ptr<hdf5DataLayerDetail::HDF5FileDataBuffer<Dtype>>
    current_buffer_;
  hsize_t current_row_;
  uint64_t offset_;
};


}  // namespace caffe

#endif  // CAFFE_HDF5_DATA_LAYER_HPP_
