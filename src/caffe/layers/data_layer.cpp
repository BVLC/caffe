#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {


template<typename Dtype, typename MItype, typename MOtype>
DataLayer<Dtype, MItype, MOtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype, MItype, MOtype>(param),
    offset_() {
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template<typename Dtype, typename MItype, typename MOtype>
DataLayer<Dtype, MItype, MOtype>::~DataLayer() {
  this->StopInternalThread();
}

template<typename Dtype, typename MItype, typename MOtype>
void DataLayer<Dtype, MItype, MOtype>::DataLayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const int_tp batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int_tp> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int_tp i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int_tp> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int_tp i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
    }
  }
  this->InitializeQuantizers(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
bool DataLayer<Dtype, MItype, MOtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype, typename MItype, typename MOtype>
void DataLayer<Dtype, MItype, MOtype>::Next() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}

// This function is called on prefetch thread
template<typename Dtype, typename MItype, typename MOtype>
void DataLayer<Dtype, MItype, MOtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0.0;
  double trans_time = 0.0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int_tp batch_size = this->layer_param_.data_param().batch_size();

  vector<Datum> datum(batch_size);

  for (int_tp item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    while (Skip()) {
      Next();
    }
    datum[item_id].ParseFromString(cursor_->value());
    if (item_id == 0) {
      // Reshape according to the first datum of each batch
      // on single input batches allows for inputs of varying dimension.
      // Use data_transformer to infer the expected blob shape from datum.
      vector<int_tp> top_shape = this->data_transformer_
          ->InferBlobShape(datum);
      this->transformed_data_.Reshape(top_shape);
      // Reshape batch according to the batch_size.
      top_shape[0] = batch_size;
      batch->data_.Reshape(top_shape);
    }
    read_time += timer.MicroSeconds();
    Next();
  }

  timer.Start();
  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = batch->label_.mutable_cpu_data();

  this->transformed_data_.set_cpu_data(top_data);
#pragma omp parallel for
  for (int_tp item_id = 0; item_id < batch_size; ++item_id) {
    // Apply data transformations (mirror, scale, crop...)
    int_tp offset = batch->data_.offset(item_id);
    this->data_transformer_->Transform(datum[item_id],
                                       &(this->transformed_data_), offset);
    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datum[item_id].label();
    }
  }
  trans_time += timer.MicroSeconds();

  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO)<< "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO)<< "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO)<< "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS_3T_GUARDED(DataLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(DataLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(DataLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(Data);
REGISTER_LAYER_CLASS_INST(Data, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(Data, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(Data, (double), (double), (double));

}  // namespace caffe
