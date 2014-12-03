#include <algorithm>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/data_sources.hpp"

namespace caffe {

template <typename Dtype>
void DataMappingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), top.size());
  for (int i = 0; i < bottom.size(); ++i) {
    CHECK_EQ(bottom[i]->channels(), 1);
    CHECK_EQ(bottom[i]->height(), 1);
    CHECK_EQ(bottom[i]->width(), 1);
  }
  const DataMappingParameter& param = this->layer_param().data_mapping_param();
  this->data_length_ = param.channels() * param.height() * param.width();
}

template <typename Dtype>
void DataMappingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    int num = bottom[i]->num();
    const Blob<Dtype>* b = bottom[i];
    Blob<Dtype>* t = top[i];
    Dtype* buffer = t->mutable_cpu_data();

    for (int j = 0; j < num; ++j) {
      index_type index = b->data_at(j, 0, 0, 0);
      Blob<Dtype>* t = top[i];
      CHECK_EQ(data_source_->retrieve(index,
                                      buffer,
                                      data_length_),
               data_length_);
      buffer += data_length_;
    }
  }
}

template <typename Dtype>
void DataMappingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), top.size());
  const DataMappingParameter& param = this->layer_param().data_mapping_param();
  for (int i = 0; i < top.size(); ++i) {
    top[i]->Reshape(bottom[i]->num(), param.channels(),
                    param.height(), param.width());
  }
}

INSTANTIATE_CLASS(DataMappingLayer);
REGISTER_LAYER_CLASS(DATA_MAPPING, DataMappingLayer);

}  // namespace caffe
