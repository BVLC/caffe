// Copyright 2014 BVLC and contributors.

#include <cuda_runtime.h>
#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <iostream>  // NOLINT(readability/streams)
#include <fstream>  // NOLINT(readability/streams)

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;
using std::pair;

namespace caffe {

template <typename Dtype>
Dtype ImageDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  CUDA_CHECK(cudaMemcpy((*top)[0]->mutable_gpu_data(),
      prefetch_data_->cpu_data(), sizeof(Dtype) * prefetch_data_->count(),
      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy((*top)[1]->mutable_gpu_data(),
      prefetch_label_->cpu_data(), sizeof(Dtype) * prefetch_label_->count(),
      cudaMemcpyHostToDevice));
  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

INSTANTIATE_CLASS(ImageDataLayer);

}  // namespace caffe
