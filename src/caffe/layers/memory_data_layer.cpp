// Copyright 2014 kloudkl@github

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
void MemoryDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "MemoryDataLayer takes two blobs as input.";
  CHECK_EQ(top->size(), 2) << "MemoryDataLayer takes two blobs as output.";
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
                     bottom[0]->height(), bottom[0]->width());
  (*top)[1]->Reshape(bottom[1]->num(), bottom[1]->channels(),
                     bottom[1]->height(), bottom[1]->width());
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  memcpy((*top)[0]->mutable_cpu_data(), bottom[0]->cpu_data(),
      sizeof(Dtype) * bottom[0]->count());
  memcpy((*top)[1]->mutable_cpu_data(), bottom[1]->cpu_data(),
      sizeof(Dtype) * bottom[1]->count());
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CUDA_CHECK(cudaMemcpy((*top)[0]->mutable_gpu_data(), bottom[0]->gpu_data(),
      sizeof(Dtype) * bottom[0]->count(), cudaMemcpyDefault
      /**< Default based unified virtual address space */));
  CUDA_CHECK(cudaMemcpy((*top)[1]->mutable_gpu_data(), bottom[1]->gpu_data(),
      sizeof(Dtype) * bottom[1]->count(), cudaMemcpyDefault));
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
