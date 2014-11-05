#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/indexed_data.hpp"

namespace caffe {

template <typename Dtype>
void IndirectionLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const IndirectionParameter& param = this->layer_param_.indirection_param();
  int size = param.source_size();
  data_length_ = param.channels() * param.height() * param.width();

  for (int i = 0; i < size; ++i) {
      shared_ptr<SimpleSingleIndexedTextFile<Dtype> > reader(
                  new SimpleSingleIndexedTextFile<Dtype>(param.source(i)));
      this->readers_.push_back(reader);
  }
}

template <typename Dtype>
void IndirectionLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>* label_blob = bottom.front();
  CHECK_EQ(label_blob->channels() * label_blob->height() * label_blob->width(),
           1) << "The input must be single labels";
  CHECK_EQ(top.size(), readers_.size())
          << "The number of top blobs must match the sources of mapping";

  const IndirectionParameter& param = this->layer_param_.indirection_param();
  int num = label_blob->num();
  int channels = param.channels();
  int height = param.height();
  int width = param.width();

  for (int i = 0; i < top.size(); ++i)
      top[i]->Reshape(num, channels, height, width);
}

template <typename Dtype>
void IndirectionLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const Dtype* labels = bottom.front()->cpu_data();
  const IndirectionParameter& param = this->layer_param_.indirection_param();

  for (int i = 0; i < top.size(); ++i) {
    Dtype* out = top[i]->mutable_cpu_data();
    int num = top[i]->num();
    for (int j = 0; j < num; ++j) {
      int index = labels[j];
      CHECK_EQ(Dtype(index), labels[j])
          << "Got non-integer as label input: " << labels[j];
      CHECK_EQ(readers_[i]->read(index, out, data_length_), data_length_)
          << "Reading data at index " << index << " failed. The source file"
             " for the mapping is " << param.source(i);
      out += data_length_;
    }
  }
}

INSTANTIATE_CLASS(IndirectionLayer);
REGISTER_LAYER_CLASS(INDIRECTION, IndirectionLayer);
}  // namespace caffe
