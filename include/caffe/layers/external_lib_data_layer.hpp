#ifndef CAFFE_EXTERNAL_LIB_DATA_LAYER_HPP_
#define CAFFE_EXTERNAL_LIB_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

class IExternalLib;
class IExternalLibDataSource;

namespace caffe {

/**
* @brief Provides data to the net from the external dynamically load library.
*/
template <typename Dtype>
class ExternalLibDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ExternalLibDataLayer(const LayerParameter& param);
  virtual ~ExternalLibDataLayer();
  virtual void DataLayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ExternalLibData"; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);
  virtual bool PrefetchToGpu(int index);

  virtual boost::shared_ptr<IExternalLib> GetExternalLib(
    const std::string& ext_lib_path,
    const std::string& factory_name,
    const std::string& ext_lib_param);

 private:
  shared_ptr<IExternalLib> external_lib_interface_;
  IExternalLibDataSource* data_source_;
  std::vector<int> gpu_prefetch_indices_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
