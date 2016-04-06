#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "ristretto/base_ristretto_layer.hpp"

namespace caffe {

template <typename Dtype>
DataRistrettoLayer<Dtype>::DataRistrettoLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param), BaseRistrettoLayer<Dtype>(),
      reader_(param) {
  this->precision_ = this->layer_param_.quantization_param().precision();
  this->rounding_ = this->layer_param_.quantization_param().rounding_scheme();
  switch (this->precision_) {
  case QuantizationParameter_Precision_FIXED_POINT:
    this->bw_layer_out_ =
        this->layer_param_.quantization_param().bw_layer_out();
    this->fl_layer_out_ =
        this->layer_param_.quantization_param().fl_layer_out();
    break;
  case QuantizationParameter_Precision_MINI_FLOATING_POINT:
    LOG(ERROR) << "DataRistrettoLayer only supports fixed point.";
    break;
  case QuantizationParameter_Precision_POWER_2_WEIGHTS:
    LOG(ERROR) << "DataRistrettoLayer only supports fixed point.";
    break;
  default:
    LOG(FATAL) << "Unknown precision mode: " << this->precision_;
    break;
  }
}

template <typename Dtype>
DataRistrettoLayer<Dtype>::~DataRistrettoLayer() {
  this->StopInternalThread();
}

// Difference to DataLayer::load_batch(): we trim the images.
// This function is called on prefetch thread
template<typename Dtype>
void DataRistrettoLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));
    // Trim data
    int batch_pixels = datum.channels() * datum.height() * datum.width();
    this->QuantizeLayerOutputs_cpu(this->transformed_data_.mutable_cpu_data(),
        batch_pixels);
    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataRistrettoLayer);
REGISTER_LAYER_CLASS(DataRistretto);

}  // namespace caffe
