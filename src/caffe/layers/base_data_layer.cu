#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  Batch<Dtype>* batch_untransformed;
  if (untransformed_top_)
    batch_untransformed = prefetch_full_untransformed_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (untransformed_top_)
    {
      top[2]->ReshapeLike(batch_untransformed->data_);
      // Copy the data
      caffe_copy(batch_untransformed->data_.count(), batch_untransformed->data_.gpu_data(),
                 top[2]->mutable_gpu_data());
    }
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CAFFE1_CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
  if (untransformed_top_)
    prefetch_free_untransformed_.push(batch_untransformed);
}

template <typename Dtype>
void BasePrefetchingSparseDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    SparseBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
    if (SparseBlob<Dtype>* sparseBlob = dynamic_cast<SparseBlob<Dtype>*>(top[0]))
    {
      // Reshape to loaded data.
      sparseBlob->ReshapeLike(batch->data_);
      // Copy the data
      caffe_copy(batch->data_.nnz(), batch->data_.gpu_data(),
         sparseBlob->mutable_gpu_data());
      caffe_copy<int>(batch->data_.nnz(), batch->data_.gpu_indices(),
         sparseBlob->mutable_gpu_indices());
      caffe_copy<int>(batch->data_.shape()[0]+1, batch->data_.gpu_ptr(),
         sparseBlob->mutable_gpu_ptr());
    } else {
    LOG(ERROR) << "The top blob in the data layer sparse is not sparse";
    LOG(FATAL) << "fatal error";
    }
 
  DLOG(INFO) << "Prefetch sparse copied (forward)";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }

  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CAFFE1_CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingSparseDataLayer);

}  // namespace caffe
