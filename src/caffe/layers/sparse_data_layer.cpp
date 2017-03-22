#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/sparse_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

#include <iostream>

namespace caffe {

template <typename Dtype>
SparseDataLayer<Dtype>::SparseDataLayer(const LayerParameter& param)
  : BasePrefetchingSparseDataLayer<Dtype>(param),
    reader_(param) {
}

// This function is used to create a thread that prefetches the data.
/*template<typename Dtype>
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
  }*/

template<typename Dtype>
SparseDataLayer<Dtype>::~SparseDataLayer<Dtype>() {
   this->StopInternalThread();
}

//TODO: DataLayerSetup
/*template <typename Dtype>
void SparseDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
					    const vector<Blob<Dtype>*>& top) {
  vector<SparseBlob<Dtype>*> sbottom;
  for (auto s: bottom)
    {
      SparseBlob<Dtype>* sb = dynamic_cast<SparseBlob<Dtype>*>(s);
      if (!sb)
	LOG(FATAL) << "DataLayerSetUp: bottom blob is not sparse";
      sbottom.push_back(sb);
    }
  vector<SparseBlob<Dtype>*> stop;
  for (auto s: top)
    {
      SparseBlob<Dtype>* sb = dynamic_cast<SparseBlob<Dtype>*>(s);
      if (!sb)
	LOG(FATAL) << "DataLayerSetUp: top blob is not sparse";
      if (sb)
	stop.push_back(sb);
      else 
	{
	  sb = new SparseBlob<Dtype>(s->label(),
	}
    }
  SparseDataLayerSetUp(sbottom,stop);
  }*/

template <typename Dtype>
void SparseDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
					    const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  //std::cerr << "\nreading top sparse point\n";
  SparseDatum& datum = *(reader_.full().peek());
  /*std::cerr << "setup datum peek label=" << datum.label() << std::endl;
  std::cerr << "output labels=" << this->output_labels_ << std::endl;
  std::cerr << "\ndone reading top sparse point\n";*/

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  //this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  if (SparseBlob<Dtype> * sparseBlob = dynamic_cast<SparseBlob<Dtype>*>(top[0])) {
    sparseBlob->Reshape(top_shape, 1);
  } else {
    LOG(ERROR)<< "The top blob in the sparse data layer is not sparse";
    LOG(FATAL) << "fatal error";
  }
  //top[0]->Reshape(top_shape);
  //std::cerr << "top shape0=" << top_shape[0] << " / top shape1=" << top_shape[1] << std::endl;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  //std::cerr << "top1 shape=" << top[1]->shape_string() << std::endl;
  LOG(INFO) << "output sparse data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  //std::cerr << "has label=" << this->output_labels_ << std::endl;
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      //std::cerr << "setup label data check=" << this->prefetch_[i].label_.data() << std::endl;
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

  /*template<typename Dtype>
  void SparseDataLayer<Dtype>::CreatePrefetchThread() {
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template<typename Dtype>
void SparseDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
  }*/

  /*template<typename Dtype>
void SparseDataLayer<Dtype>::LayerSetUp(const vector<SparseBlob<Dtype>*>& bottom,
					const vector<SparseBlob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  SparseDatum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
  }*/

  /*template<typename Dtype>
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
  }*/

  /*template<typename Dtype>
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
  }*/

// This function is called on prefetch thread
template<typename Dtype>
void SparseDataLayer<Dtype>::load_batch(SparseBatch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  //double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  //CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  SparseDatum& pdatum = *(reader_.full().peek());
  //std::cerr << "peek datum label=" << pdatum.label() << std::endl;
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(pdatum);

  //this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  //std::cerr << "sparse top_shape=" << top_shape[0] << " / " << top_shape[1] << std::endl;
  
  vector<SparseDatum*> datums;
  //vector<int> labels;
  for (int item_id = 0; item_id < batch_size; item_id++) {
    timer.Start();
    // get a datum
    SparseDatum *datum = reader_.full().pop("Waiting for sparse data");
    //std::cerr << "item_id=" << item_id << " / datum label=" << datum->label() << std::endl;
    read_time += timer.MicroSeconds();
    /*if (this->output_labels_)
      labels.push_back(datum.label());*/
    datums.push_back(datum);
    timer.Start();
    //trans_time += timer.MicroSeconds();
    //reader_.free().push(const_cast<SparseDatum*>(datum));
  }
  
  int nnz = 0;
  for (int i = 0; i < batch_size; i++) {
    nnz += datums[i]->nnz();
  }
  //std::cerr << "\nload_batch batch nnz=" << nnz << std::endl;
  batch->data_.Reshape(top_shape,nnz);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  int* indices = batch->data_.mutable_cpu_indices();
  int* ptr = batch->data_.mutable_cpu_ptr();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  ptr[0] = 0;
  int pos = 0;
  for (int item_id = 0; item_id < batch_size; item_id++) {
    SparseDatum* datum = datums.at(item_id);
    for (int k=0;k<datum->nnz();k++) {
      top_data[k + pos] = datum->data(k);
      indices[k + pos] = datum->indices(k);
    }
    pos += datum->nnz();
    ptr[item_id + 1] = pos;
    
    // Copy label.
    //std::cerr << "load batch output labels=" << this->output_labels_ << std::endl;
    if (this->output_labels_) {
      //std::cerr << "loading label=" << item_id << " / " << datum->label() << std::endl;;
      top_label[item_id] = datum->label();
    }
    reader_.free().push(const_cast<SparseDatum*>(datum));
  }

  //debug
  /*std::cerr << "indices size=" << nnz << " / indices=";
  for (int ind=0;ind<nnz;ind++)
    std::cerr << indices[ind] << " ";
    std::cerr << std::endl;*/
  //debug

  /*for (int item_id = 0; item_id < batch_size; ++item_id) {
    datum = datums.at(item_id);
    reader_.free().push(const_cast<SparseDatum*>(&datum));
    }*/
  
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch sparse batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  //DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

  /*#ifdef CPU_ONLY
STUB_GPU_FORWARD(SparseDataLayer, Forward);
#endif*/

INSTANTIATE_CLASS(SparseDataLayer);
REGISTER_LAYER_CLASS(SparseData);

}  // namespace caffe
