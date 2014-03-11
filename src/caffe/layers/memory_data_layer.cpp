// Copyright 2014 kloudkl@github

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
MemoryDataLayer<Dtype>::MemoryDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param), num_data_blobs_(param.datum_dims_size()) {
  for (int i = 0; i < param.datum_dims_size(); ++i) {
    datum_dims_.push_back(param.datum_dims(i));
  }
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), num_data_blobs_) <<
      "MemoryDataLayer takes " << num_data_blobs_ << " blobs as input.";
  CHECK_EQ(top->size(), num_data_blobs_) <<
      "MemoryDataLayer takes " << num_data_blobs_ << " blobs as output.";
  for (int i = 0; i < num_data_blobs_; ++i) {
    CHECK_EQ(bottom[i]->channels(), datum_dims_[i].channels());
    CHECK_EQ(bottom[i]->height(), datum_dims_[i].height());
    CHECK_EQ(bottom[i]->width(), datum_dims_[i].width());
    (*top)[i]->Reshape(bottom[i]->num(), datum_dims_[i].channels(),
                       datum_dims_[i].height(), datum_dims_[i].width());
  }
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < num_data_blobs_; ++i) {
    memcpy((*top)[i]->mutable_cpu_data(), bottom[i]->cpu_data(),
        sizeof(Dtype) * bottom[i]->count());
  }
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < num_data_blobs_; ++i) {
    CUDA_CHECK(cudaMemcpy((*top)[i]->mutable_gpu_data(), bottom[i]->gpu_data(),
        sizeof(Dtype) * bottom[i]->count(), cudaMemcpyDefault
        /**< Default based unified virtual address space */));
  }
}

template <typename Dtype>
Dtype MemoryDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

template <typename Dtype>
Dtype MemoryDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(MemoryDataLayer);

}  // namespace caffe
