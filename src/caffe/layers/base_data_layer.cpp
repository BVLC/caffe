#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
}



INSTANTIATE_CLASS(BaseDataLayer);

}  // namespace caffe
