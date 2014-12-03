#include <algorithm>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/data_sources.hpp"

namespace caffe {

template <typename Dtype>
void DataSequenceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 0);
  CHECK_EQ(top.size(), 1);
  const DataSequenceParameter& param =
      this->layer_param().data_sequence_param();
  this->data_length_ = param.channels() * param.height() * param.width();
  this->indices_ = data_source_->indices();
  std::sort(indices_.begin(), indices_.end());
  cursor_ = -1;
  batch_size_ = param.batch_size();
}

template <typename Dtype>
void DataSequenceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  Dtype* buffer = top[0]->mutable_cpu_data();
  for (int i = 0; i < batch_size_; ++i) {
    cursor_ = (cursor_ + 1) % indices_.size();
    CHECK_EQ(data_source_->retrieve(indices_[cursor_], buffer, data_length_),
             data_length_);
    buffer += data_length_;
  }
}

template <typename Dtype>
void DataSequenceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  const DataSequenceParameter& param =
      this->layer_param().data_sequence_param();
  top[0]->Reshape(batch_size_, param.channels(),
                  param.height(), param.width());
}

INSTANTIATE_CLASS(DataSequenceLayer);
REGISTER_LAYER_CLASS(DATA_SEQUENCE, DataSequenceLayer);

}  // namespace caffe

