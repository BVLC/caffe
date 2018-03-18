#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
BaseDataLayer<Dtype, MItype, MOtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype, MItype, MOtype>(param),
      transform_param_(param.transform_param()) {
}

template<typename Dtype, typename MItype, typename MOtype>
void BaseDataLayer<Dtype, MItype, MOtype>::LayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_,
                                 this->phase_, this->device_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
  this->InitializeQuantizers(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
BasePrefetchingDataLayer<Dtype, MItype, MOtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype, MItype, MOtype>(param),
      prefetch_(param.data_param().prefetch()),
      prefetch_free_(), prefetch_full_(), prefetch_current_() {
  for (int i = 0; i < prefetch_.size(); ++i) {
    prefetch_[i].reset(new Batch<Dtype>());
    prefetch_free_.push(prefetch_[i].get());
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void BasePrefetchingDataLayer<Dtype, MItype, MOtype>::LayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  BaseDataLayer<Dtype, MItype, MOtype>::LayerSetUp(bottom, top);

  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int_tp i = 0; i < prefetch_.size(); ++i) {
    prefetch_[i]->data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i]->label_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int_tp i = 0; i < prefetch_.size(); ++i) {
      prefetch_[i]->data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i]->label_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread(this->get_device());
  DLOG(INFO) << "Prefetch initialized.";
}

template<typename Dtype, typename MItype, typename MOtype>
void BasePrefetchingDataLayer<Dtype, MItype, MOtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
#ifdef USE_CUDA
  cudaStream_t stream;
  if (this->get_device()->backend() == BACKEND_CUDA) {
    if (Caffe::mode() == Caffe::GPU) {
      if (this->get_device()->backend() == BACKEND_CUDA) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      }
    }
  }
#endif  // USE_CUDA
#endif  // !CPU_ONLY

  try {
    while (!must_stop()) {
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
#ifdef USE_CUDA
      if (Caffe::mode() == Caffe::GPU) {
        // FIXME: Async load
        /*
        if (this->get_device()->backend() == BACKEND_CUDA) {
          batch->data_.data().get()->async_gpu_push(stream);
          if (this->output_labels_) {
            batch->label_.data().get()->async_gpu_push(stream);
          }
          CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        */
      }
#endif  // USE_CUDA
#endif  // !CPU_ONLY
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
#ifdef USE_CUDA
  if (Caffe::mode() == Caffe::GPU) {
    if (this->get_device()->backend() == BACKEND_CUDA) {
      CUDA_CHECK(cudaStreamDestroy(stream));
    }
  }
#endif  // USE_CUDA
#endif  // !CPU_ONLY
}

template<typename Dtype, typename MItype, typename MOtype>
void BasePrefetchingDataLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS_3T_GUARDED(BaseDataLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(BaseDataLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(BaseDataLayer, (double), (double), (double));
INSTANTIATE_CLASS_3T_GUARDED(BaseDataLayer, (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASS_3T_GUARDED(BaseDataLayer, (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASS_3T_GUARDED(BaseDataLayer, (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASS_3T_GUARDED(BaseDataLayer, (uint64_t), (uint64_t), (uint64_t));

INSTANTIATE_CLASS_3T_GUARDED(BasePrefetchingDataLayer,
                             (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(BasePrefetchingDataLayer,
                             (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(BasePrefetchingDataLayer,
                             (double), (double), (double));
INSTANTIATE_CLASS_3T_GUARDED(BasePrefetchingDataLayer,
                             (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASS_3T_GUARDED(BasePrefetchingDataLayer,
                             (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASS_3T_GUARDED(BasePrefetchingDataLayer,
                             (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASS_3T_GUARDED(BasePrefetchingDataLayer,
                             (uint64_t), (uint64_t), (uint64_t));
}  // namespace caffe
