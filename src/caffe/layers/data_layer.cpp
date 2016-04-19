#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
#ifdef _OPENMP
  previous_batch_size_ = batch_size;
  this->transformed_datas_.resize(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    this->transformed_datas_[i].reset(new Blob<Dtype>(top_shape));
  }
#endif
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
}

// This function is called on prefetch thread
template<typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CPUTimer trans_timer;
  CHECK(batch->data_.count());

// For transformed_datas_ we check only the first one
#ifdef _OPENMP
  CHECK(this->transformed_datas_[0].get()->count());
#else
  CHECK(this->transformed_data_.count());
#endif

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
#ifdef _OPENMP
  if (batch_size != this->previous_batch_size_) {
    this->transformed_datas_.resize(batch_size);
    // deallocate redundant blobs
    for (int i = previous_batch_size_; i < batch_size; ++i) {
      this->transformed_datas_[i].reset(new Blob<Dtype>());
    }
    this->previous_batch_size_ = batch_size;
  }

  for (int i = 0; i< this->transformed_datas_.size(); ++i) {
    this->transformed_datas_[i]->Reshape(top_shape);  // TODO: investigate
  }
#else
  this->transformed_data_.Reshape(top_shape);
#endif
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  trans_timer.Start();
#ifdef _OPENMP
  #pragma omp parallel
  #pragma omp single nowait
#endif
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    timer.Stop();
    read_time += timer.MicroSeconds();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);

#ifdef _OPENMP
    this->transformed_datas_[item_id]->set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum,
                                       this->transformed_datas_[item_id].get());
#else
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));
#endif
    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    reader_.free().push(const_cast<Datum*>(&datum));
  }
  trans_timer.Stop();
  batch_timer.Stop();
  // Due to multithreaded nature of transformation,
  // time it takes to execute them we get from subtracting
  // read batch of images time from total batch read&transform time
  trans_time = trans_timer.MicroSeconds() - read_time;
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
