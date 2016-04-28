#include <leveldb/db.h>
#include <pthread.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

namespace caffe {

template<typename Dtype>
void SparseDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  prefetch_data_.swap(prefetch_data_copy_);
  prefetch_label_.swap(prefetch_label_copy_);

  // Start a new prefetch thread
  CreatePrefetchThread();

  if (SparseBlob<Dtype> * sparseBlob =
      dynamic_cast<SparseBlob<Dtype>*>(top[0])) {
    sparseBlob->set_gpu_data(
        const_cast<Dtype*>(prefetch_data_copy_->gpu_data()),
        const_cast<int*>(prefetch_data_copy_->gpu_indices()),
        const_cast<int*>(prefetch_data_copy_->gpu_ptr()),
        prefetch_data_copy_->nnz(), prefetch_data_copy_->nnz());
  } else {
    LOG(FATAL)<< "The top blob in the data layer sparse is not sparse\n";
  }

  if (output_labels_) {
    caffe_copy(prefetch_label_copy_->count(), prefetch_label_copy_->cpu_data(),
               top[1]->mutable_gpu_data());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SparseDataLayer);

}  // namespace caffe
