<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
#include <boost/thread.hpp>
#include <vector>

<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/data_layers.hpp"
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
=======
>>>>>>> pod/device/blob.hpp
=======
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
#include <boost/thread.hpp>
#include <vector>

#include "caffe/data_layers.hpp"
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/util/io.hpp"
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> BVLC/master
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
<<<<<<< HEAD
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include <boost/thread.hpp>
#include <vector>

#include "caffe/data_layers.hpp"
>>>>>>> device-abstraction
=======
#include <boost/thread.hpp>
#include <vector>

#include "caffe/data_layers.hpp"
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
      transform_param_(param.transform_param()) {
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
      transform_param_(param.transform_param()) {
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
      transform_param_(param.transform_param()) {
=======
>>>>>>> pod/device/blob.hpp
=======
      transform_param_(param.transform_param()) {
=======
>>>>>>> pod/device/blob.hpp
      transform_param_(param.transform_param()),
      data_transformer_(transform_param_) {
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
      transform_param_(param.transform_param()) {
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
      transform_param_(param.transform_param()) {
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
      transform_param_(param.transform_param()) {
=======
      transform_param_(param.transform_param()),
      data_transformer_(transform_param_) {
>>>>>>> origin/BVLC/parallel
=======
      transform_param_(param.transform_param()) {
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
      transform_param_(param.transform_param()) {
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
      transform_param_(param.transform_param()) {
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
=======
      transform_param_(param.transform_param()) {
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
      transform_param_(param.transform_param()) {
>>>>>>> device-abstraction
=======
      transform_param_(param.transform_param()) {
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
=======
>>>>>>> caffe
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
<<<<<<< HEAD
=======
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
  data_transformer_.InitRand();
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
  data_transformer_.InitRand();
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> device-abstraction
      prefetch_free_(), prefetch_full_() {
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
>>>>>>> pod/device/blob.hpp
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
=======
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
<<<<<<< HEAD
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
=======
>>>>>>> pod/caffe-merge
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
=======
=======

>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======

>>>>>>> origin/BVLC/parallel
>>>>>>> pod/common.hpp
=======
<<<<<<< HEAD

>>>>>>> origin/BVLC/parallel
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======

>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======

>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge

>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======

>>>>>>> origin/BVLC/parallel
=======
>>>>>>> device-abstraction
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
=======
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
=======
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/common.hpp
=======
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/common.hpp
=======
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
=======
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
=======
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> device-abstraction
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i].label_.mutable_cpu_data();
    }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  }
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
  }
=======
  }
=======
  }
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  }
>>>>>>> device-abstraction
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i].label_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  }
=======
  }
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  }
>>>>>>> origin/BVLC/parallel
  switch (Caffe::mode()) {
    case Caffe::CPU:
      device_ = -1;
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      for(int i = 0; i < PREFETCH_COUNT; ++i) {
        prefetch_[i].data_.mutable_gpu_data();
        if (this->output_labels_) {
          prefetch_[i].label_.mutable_gpu_data();
        }
      }
      CUDA_CHECK(cudaGetDevice(&device_));
#endif
      break;
  }
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp

  DLOG(INFO) << "Initializing prefetch";
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/common.hpp
  }
=======
  }
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
  switch (Caffe::mode()) {
    case Caffe::CPU:
      device_ = -1;
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      for(int i = 0; i < PREFETCH_COUNT; ++i) {
        prefetch_[i].data_.mutable_gpu_data();
        if (this->output_labels_) {
          prefetch_[i].label_.mutable_gpu_data();
        }
      }
      CUDA_CHECK(cudaGetDevice(&device_));
#endif
      break;
  }

<<<<<<< HEAD
  DLOG(INFO) << "Initializing prefetch";
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> device-abstraction
  DLOG(INFO) << "Prefetch initialized.";
=======
template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
<<<<<<< HEAD
      transform_param_(param.transform_param()) {
=======
<<<<<<< HEAD
<<<<<<< HEAD
      transform_param_(param.transform_param()) {
=======
      transform_param_(param.transform_param()),
      data_transformer_(transform_param_) {
>>>>>>> origin/BVLC/parallel
=======
      transform_param_(param.transform_param()) {
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
}

template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
<<<<<<< HEAD
#endif

  try {
    while (!must_stop()) {
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
BasePrefetchingDataLayer<Dtype>::~BasePrefetchingDataLayer() {
  CHECK(StopInternalThread()) << "Stop thread failed";
=======
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
=======
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
  data_transformer_.InitRand();
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
}

template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
  }
=======
=======
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/post-rebase-error-fix
      prefetch_free_(), prefetch_full_() {
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
=======
>>>>>>> pod/device/blob.hpp
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
=======
>>>>>>> pod/device/blob.hpp
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
>>>>>>> pod/device/blob.hpp
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
=======
<<<<<<< HEAD
=======
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD

>>>>>>> origin/BVLC/parallel
>>>>>>> pod/common.hpp
=======
<<<<<<< HEAD

>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
=======
<<<<<<< HEAD
=======
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======

>>>>>>> origin/BVLC/parallel
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/common.hpp
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
=======
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> caffe
=======
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
=======
=======
>>>>>>> pod/device/blob.hpp
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> pod-caffe-pod.hpp-merge
=======
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
=======
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
=======
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> device-abstraction
=======
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> pod/post-rebase-error-fix
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i].label_.mutable_cpu_data();
    }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  }
=======
  }
<<<<<<< HEAD
=======
  }
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
  }
>>>>>>> device-abstraction
=======
  }
>>>>>>> pod/post-rebase-error-fix
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i].label_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
}
>>>>>>> pod/device/blob.hpp

>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
  switch (Caffe::mode()) {
    case Caffe::CPU:
      device_ = -1;
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      for(int i = 0; i < PREFETCH_COUNT; ++i) {
        prefetch_[i].data_.mutable_gpu_data();
        if (this->output_labels_) {
          prefetch_[i].label_.mutable_gpu_data();
        }
      }
      CUDA_CHECK(cudaGetDevice(&device_));
#endif
      break;
  }
<<<<<<< HEAD

  DLOG(INFO) << "Initializing prefetch";
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
<<<<<<< HEAD
<<<<<<< HEAD
=======
  }
  switch (Caffe::mode()) {
    case Caffe::CPU:
      device_ = -1;
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      for(int i = 0; i < PREFETCH_COUNT; ++i) {
        prefetch_[i].data_.mutable_gpu_data();
        if (this->output_labels_) {
          prefetch_[i].label_.mutable_gpu_data();
        }
      }
      CUDA_CHECK(cudaGetDevice(&device_));
#endif
      break;
  }

  DLOG(INFO) << "Initializing prefetch";
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
>>>>>>> origin/BVLC/parallel
  DLOG(INFO) << "Prefetch initialized.";
=======
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
  data_transformer_.InitRand();
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
      prefetch_free_(), prefetch_full_() {
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
<<<<<<< HEAD
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
=======
<<<<<<< HEAD
=======
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
}
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp

template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
BasePrefetchingDataLayer<Dtype>::~BasePrefetchingDataLayer() {
  CHECK(StopInternalThread()) << "Stop thread failed";
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======

>>>>>>> origin/BVLC/parallel
>>>>>>> pod/device/blob.hpp
=======

>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> pod/caffe-merge
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> caffe
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
=======
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i].label_.mutable_cpu_data();
    }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/common.hpp
  }
  switch (Caffe::mode()) {
    case Caffe::CPU:
      device_ = -1;
      break;
    case Caffe::GPU:
=======
<<<<<<< HEAD
  }
=======
  }
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
#ifndef CPU_ONLY
      for(int i = 0; i < PREFETCH_COUNT; ++i) {
        prefetch_[i].data_.mutable_gpu_data();
        if (this->output_labels_) {
          prefetch_[i].label_.mutable_gpu_data();
        }
      }
      CUDA_CHECK(cudaGetDevice(&device_));
#endif
      break;
  }

  DLOG(INFO) << "Initializing prefetch";
<<<<<<< HEAD
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
  this->data_transformer_->InitRand();
  StartInternalThread();
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
  }
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
  switch (Caffe::mode()) {
    case Caffe::CPU:
      device_ = -1;
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      for(int i = 0; i < PREFETCH_COUNT; ++i) {
        prefetch_[i].data_.mutable_gpu_data();
        if (this->output_labels_) {
          prefetch_[i].label_.mutable_gpu_data();
        }
      }
      CUDA_CHECK(cudaGetDevice(&device_));
#endif
      break;
  }

  DLOG(INFO) << "Initializing prefetch";
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
  DLOG(INFO) << "Prefetch initialized.";
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> caffe
=======
=======
  }
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
  switch (Caffe::mode()) {
    case Caffe::CPU:
      device_ = -1;
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      for(int i = 0; i < PREFETCH_COUNT; ++i) {
        prefetch_[i].data_.mutable_gpu_data();
        if (this->output_labels_) {
          prefetch_[i].label_.mutable_gpu_data();
        }
      }
      CUDA_CHECK(cudaGetDevice(&device_));
#endif
      break;
  }

  DLOG(INFO) << "Initializing prefetch";
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
  }
  switch (Caffe::mode()) {
    case Caffe::CPU:
      device_ = -1;
      break;
    case Caffe::GPU:
=======
  }
=======
  }
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
#ifndef CPU_ONLY
      for(int i = 0; i < PREFETCH_COUNT; ++i) {
        prefetch_[i].data_.mutable_gpu_data();
        if (this->output_labels_) {
          prefetch_[i].label_.mutable_gpu_data();
        }
      }
      CUDA_CHECK(cudaGetDevice(&device_));
#endif
      break;
  }

  DLOG(INFO) << "Initializing prefetch";
<<<<<<< HEAD
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
  this->data_transformer_->InitRand();
  StartInternalThread();
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
  }
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
  switch (Caffe::mode()) {
    case Caffe::CPU:
      device_ = -1;
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      for(int i = 0; i < PREFETCH_COUNT; ++i) {
        prefetch_[i].data_.mutable_gpu_data();
        if (this->output_labels_) {
          prefetch_[i].label_.mutable_gpu_data();
        }
      }
      CUDA_CHECK(cudaGetDevice(&device_));
#endif
      break;
  }

  DLOG(INFO) << "Initializing prefetch";
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
  }
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
  switch (Caffe::mode()) {
    case Caffe::CPU:
      device_ = -1;
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      for(int i = 0; i < PREFETCH_COUNT; ++i) {
        prefetch_[i].data_.mutable_gpu_data();
        if (this->output_labels_) {
          prefetch_[i].label_.mutable_gpu_data();
        }
      }
      CUDA_CHECK(cudaGetDevice(&device_));
#endif
      break;
  }

  DLOG(INFO) << "Initializing prefetch";
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
  DLOG(INFO) << "Prefetch initialized.";
>>>>>>> pod-caffe-pod.hpp-merge
}

template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
=======
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
>>>>>>> device-abstraction
=======
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
>>>>>>> pod/post-rebase-error-fix
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
  }
  switch (Caffe::mode()) {
    case Caffe::CPU:
      device_ = -1;
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      for(int i = 0; i < PREFETCH_COUNT; ++i) {
        prefetch_[i].data_.mutable_gpu_data();
        if (this->output_labels_) {
          prefetch_[i].label_.mutable_gpu_data();
        }
      }
      CUDA_CHECK(cudaGetDevice(&device_));
#endif
      break;
  }

  DLOG(INFO) << "Initializing prefetch";
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
=======
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
BasePrefetchingDataLayer<Dtype>::~BasePrefetchingDataLayer() {
  CHECK(StopInternalThread()) << "Stop thread failed";
}

template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/common.hpp
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/common.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/common.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
BasePrefetchingDataLayer<Dtype>::~BasePrefetchingDataLayer() {
  CHECK(StopInternalThread()) << "Stop thread failed";
}

template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if(device_ >= 0) {
    CUDA_CHECK(cudaSetDevice(device_));
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }
#endif

  while(!must_stop()) {
    Batch<Dtype>* batch = prefetch_free_.pop();
    load_batch(batch);
#ifndef CPU_ONLY
    if(device_ >= 0) {
      batch->data_.data().get()->async_gpu_push(stream);
      cudaStreamSynchronize(stream);
    }
#endif
    prefetch_full_.push(batch);
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
=======
>>>>>>> pod/common.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
BasePrefetchingDataLayer<Dtype>::~BasePrefetchingDataLayer() {
  CHECK(StopInternalThread()) << "Stop thread failed";
}

template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if(device_ >= 0) {
    CUDA_CHECK(cudaSetDevice(device_));
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }
#endif

  while(!must_stop()) {
    Batch<Dtype>* batch = prefetch_free_.pop();
    load_batch(batch);
#ifndef CPU_ONLY
    if(device_ >= 0) {
      batch->data_.data().get()->async_gpu_push(stream);
      cudaStreamSynchronize(stream);
    }
#endif
    prefetch_full_.push(batch);
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/common.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/common.hpp
=======
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
>>>>>>> device-abstraction
=======
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
>>>>>>> pod/post-rebase-error-fix
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======

  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
      top[0]->mutable_cpu_data());
  if (this->output_labels_) {
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD

  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
      top[0]->mutable_cpu_data());
  if (this->output_labels_) {
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD

  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
      top[0]->mutable_cpu_data());
  if (this->output_labels_) {
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD

  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
      top[0]->mutable_cpu_data());
  if (this->output_labels_) {
>>>>>>> origin/BVLC/parallel
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp

  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
      top[0]->mutable_cpu_data());
  if (this->output_labels_) {
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp

  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
      top[0]->mutable_cpu_data());
  if (this->output_labels_) {
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD

  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
      top[0]->mutable_cpu_data());
  if (this->output_labels_) {
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD

  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
      top[0]->mutable_cpu_data());
  if (this->output_labels_) {
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======

  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
      top[0]->mutable_cpu_data());
  if (this->output_labels_) {
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
      prefetch_free_(), prefetch_full_() {
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
<<<<<<< HEAD
=======
      prefetch_free_(), prefetch_full_(), device_() {
  for(int i = 0; i < PREFETCH_COUNT; ++i)
    prefetch_free_.push(&prefetch_[i]);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
<<<<<<< HEAD
<<<<<<< HEAD
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
=======
  for(int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> origin/BVLC/parallel
=======
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
>>>>>>> caffe
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i].label_.mutable_cpu_data();
    }
  }
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i].label_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
<<<<<<< HEAD
=======
  switch (Caffe::mode()) {
    case Caffe::CPU:
      device_ = -1;
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      for(int i = 0; i < PREFETCH_COUNT; ++i) {
        prefetch_[i].data_.mutable_gpu_data();
        if (this->output_labels_) {
          prefetch_[i].label_.mutable_gpu_data();
        }
      }
      CUDA_CHECK(cudaGetDevice(&device_));
#endif
      break;
  }

  DLOG(INFO) << "Initializing prefetch";
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
<<<<<<< HEAD
=======
BasePrefetchingDataLayer<Dtype>::~BasePrefetchingDataLayer() {
  CHECK(StopInternalThread()) << "Stop thread failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if(device_ >= 0) {
    CUDA_CHECK(cudaSetDevice(device_));
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }
#endif

  while(!must_stop()) {
    Batch<Dtype>* batch = prefetch_free_.pop();
    load_batch(batch);
#ifndef CPU_ONLY
    if(device_ >= 0) {
      batch->data_.data().get()->async_gpu_push(stream);
      cudaStreamSynchronize(stream);
    }
#endif
    prefetch_full_.push(batch);
  }
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======

  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
      top[0]->mutable_cpu_data());
  if (this->output_labels_) {
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());
  }

  prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
INSTANTIATE_CLASS(Batch);
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
INSTANTIATE_CLASS(Batch);
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
INSTANTIATE_CLASS(Batch);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
INSTANTIATE_CLASS(Batch);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
INSTANTIATE_CLASS(Batch);
=======
INSTANTIATE_CLASS(Batch);
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
INSTANTIATE_CLASS(Batch);
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
INSTANTIATE_CLASS(Batch);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
INSTANTIATE_CLASS(Batch);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
