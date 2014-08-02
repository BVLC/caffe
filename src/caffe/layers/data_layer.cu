// Copyright 2014 BVLC and contributors.

#include <string>
#include <vector>

#include "leveldb/db.h"
#include "pthread.h"
#include "stdint.h"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
Dtype DataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      (*top)[0]->mutable_gpu_data());
  if (output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        (*top)[1]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

INSTANTIATE_CLASS(DataLayer);

}  // namespace caffe
