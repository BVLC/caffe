#include <pthread.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/sparse_blob.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

// This function is used to create a thread that prefetches the data.
template<typename Dtype>
void SparseDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  CPUTimer timer;

  CHECK(prefetch_data_->count());
  CHECK(prefetch_data_copy_->count());

  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (output_labels_) {
    top_label = prefetch_label_->mutable_cpu_data();
  }
  const int batch_size =
      this->layer_param_.sparse_data_param().batch_size();
  const int size = this->datum_size_;
  vector<shared_ptr<SparseDatum> > datums;
  timer.Start();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // TODO can we get rid of this copy
    shared_ptr<SparseDatum> datum( new SparseDatum());
    datum->ParseFromString(cursor_->value());
    datums.push_back(datum);
    if (output_labels_) {
      top_label[item_id] = datum->label();
    }
    // go to the next iter
    cursor_->Next();
    if (!cursor_->valid()) {
      DLOG(INFO)<< "Restarting data prefetching from start.";
      cursor_->SeekToFirst();
    }
  }
  double read_time = timer.MicroSeconds();
  timer.Start();
  int nnz = 0;
  for (int i = 0; i < batch_size; i++) {
    nnz += datums[i]->nnz();
  }
  vector<int> shape_vec(2);
  shape_vec[0] = batch_size;
  shape_vec[1] = size;
  prefetch_data_->Reshape(shape_vec, nnz);

  Dtype* top_data = prefetch_data_->mutable_cpu_data();
  int* indices = prefetch_data_->mutable_cpu_indices();
  int* ptr = prefetch_data_->mutable_cpu_ptr();

  ptr[0] = 0;
  int pos = 0;
  for (int i = 0; i < batch_size; i++) {
    shared_ptr<SparseDatum> d = datums[i];
    for (int k = 0; k < d->nnz(); k++) {
      top_data[k + pos] = d->data(k);
      indices[k + pos] = d->indices(k);
    }
    pos += d->nnz();
    ptr[i + 1] = pos;
  }
  double write_time = timer.MicroSeconds();

  batch_timer.Stop();
  DLOG(INFO)<< "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO)<< "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO)<< "Write time: " << write_time / 1000 << " ms.";
}

template<typename Dtype>
SparseDataLayer<Dtype>::~SparseDataLayer<Dtype>() {
  JoinPrefetchThread();
}

template<typename Dtype>
void SparseDataLayer<Dtype>::CreatePrefetchThread() {
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template<typename Dtype>
void SparseDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template<typename Dtype>
void SparseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }

  // Initialize DB
  db_.reset(db::GetDB(this->layer_param_.sparse_data_param().backend()));
  db_->Open(this->layer_param_.sparse_data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());

  // Check if we should randomly skip a few data points
  if (this->layer_param_.sparse_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand()
        % this->layer_param_.sparse_data_param().rand_skip();
    LOG(INFO)<< "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      cursor_->Next();
    }
  }
  // Read a data point, and use it to initialize the top blob.
  SparseDatum datum;
  datum.ParseFromString(cursor_->value());

  vector<int> shape_vec(2);
  shape_vec[0] = this->layer_param_.sparse_data_param().batch_size();
  shape_vec[1] = datum.size();

  if (SparseBlob<Dtype> * sparseBlob = dynamic_cast<SparseBlob<Dtype>*>(
      top[0])) {
    sparseBlob->Reshape(shape_vec, 1);
  } else {
    LOG(FATAL)<< "The top blob in the data layer sparse is not sparse\n";
  }
  LOG(INFO)<< "size of shape_vec in test: " << shape_vec.size();
  prefetch_data_.reset(
      new SparseBlob<Dtype>(shape_vec, 1));
  prefetch_data_copy_.reset(
      new SparseBlob<Dtype>(shape_vec, 1));

  LOG(INFO)<< "output data size: " << top[0]->num() << ","
  << top[0]->channels() << "," << top[0]->height() << ","
  << top[0]->width();
  // label
  if (output_labels_) {
    vector<int> shape_label(
        1, this->layer_param_.sparse_data_param().batch_size());
    top[1]->Reshape(shape_label);
    this->prefetch_label_.reset(
        new Blob<Dtype>(shape_label));
    prefetch_label_copy_.reset(
        new Blob<Dtype>(shape_label));
  }
  datum_size_ = datum.size();

  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  if (output_labels_) {
    prefetch_label_->mutable_cpu_data();
  }
  DLOG(INFO)<< "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO)<< "Prefetch initialized.";
}

template<typename Dtype>
void SparseDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // we swap the prefetch data
  prefetch_data_.swap(prefetch_data_copy_);
  prefetch_label_.swap(prefetch_label_copy_);

  // Start a new prefetch thread ahead of any memory transfer
  CreatePrefetchThread();

  if (SparseBlob<Dtype> * sparseBlob = dynamic_cast<SparseBlob<Dtype>*>(
      top[0])) {
    sparseBlob->set_cpu_data(
        const_cast<Dtype*>(prefetch_data_copy_->cpu_data()),
        const_cast<int*>(prefetch_data_copy_->cpu_indices()),
        const_cast<int*>(prefetch_data_copy_->cpu_ptr()),
        prefetch_data_copy_->nnz(), prefetch_data_copy_->nnz());
  } else {
    LOG(FATAL)<< "The top blob in the data layer sparse is not sparse\n";
  }
  if (output_labels_) {
    caffe_copy(prefetch_label_copy_->count(), prefetch_label_copy_->cpu_data(),
               top[1]->mutable_cpu_data());
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(SparseDataLayer, Forward);
#endif

INSTANTIATE_CLASS(SparseDataLayer);
REGISTER_LAYER_CLASS(SparseData);

}  // namespace caffe
