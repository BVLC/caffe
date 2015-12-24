#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>
#include <string>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
CPMDataLayer<Dtype>::CPMDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param, true) {
}

template <typename Dtype>
CPMDataLayer<Dtype>::~CPMDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void CPMDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());
  LOG(INFO) << datum.height() << " " << datum.width() << " " << datum.channels();

  bool force_color = this->layer_param_.cpmdata_param().force_encoded_color();
  if ((force_color && DecodeDatum(&datum, true)) ||
      DecodeDatumNative(&datum)) {
    LOG(INFO) << "Decoding Datum";
  }

  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.cpmdata_param().batch_size();
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
    }
    this->transformed_data_.Reshape(1, datum.channels(), crop_size, crop_size);
  } 
  else {
    const int height = this->phase_ != TRAIN ? datum.height() :
      this->layer_param_.transform_param().crop_size_y();
    const int width = this->phase_ != TRAIN ? datum.width() :
      this->layer_param_.transform_param().crop_size_x();
    LOG(INFO) << "PREFETCH_COUNT is " << this->PREFETCH_COUNT;
    top[0]->Reshape(batch_size, datum.channels(), height, width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, datum.channels(), height, width);
    }
    this->transformed_data_.Reshape(1, 4, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // label
  if (this->output_labels_) {
    const int stride = this->layer_param_.transform_param().stride();
    const int height = this->phase_ != TRAIN ? datum.height() :
      this->layer_param_.transform_param().crop_size_y();
    const int width = this->phase_ != TRAIN ? datum.width() :
      this->layer_param_.transform_param().crop_size_x();

    int num_parts = this->layer_param_.transform_param().num_parts();
    top[1]->Reshape(batch_size, 2*(num_parts+1), height/stride, width/stride);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(batch_size, 2*(num_parts+1), height/stride, width/stride);
    }
    this->transformed_label_.Reshape(1, 2*(num_parts+1), height/stride, width/stride);
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void CPMDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

  CPUTimer batch_timer;
  batch_timer.Start();
  double deque_time = 0;
  double decod_time = 0;
  double trans_time = 0;
  static int cnt = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape on single input batches for inputs of varying dimension.
  const int batch_size = this->layer_param_.cpmdata_param().batch_size();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  bool force_color = this->layer_param_.cpmdata_param().force_encoded_color();
  if (batch_size == 1 && crop_size == 0) {
    Datum& datum = *(reader_.full().peek());
    if (datum.encoded()) {
      if (force_color) {
        DecodeDatum(&datum, true);
      } else {
        DecodeDatumNative(&datum);
      }
    }
    batch->data_.Reshape(1, datum.channels(),
        datum.height(), datum.width());
        this->transformed_data_.Reshape(1, datum.channels(),
        datum.height(), datum.width());
  }

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    deque_time += timer.MicroSeconds();

    timer.Start();
    cv::Mat cv_img;
    if (datum.encoded()) {
      if (force_color) {
        cv_img = DecodeDatumToCVMat(datum, true);
      } else {
        cv_img = DecodeDatumToCVMatNative(datum);
      }
      if (cv_img.channels() != this->transformed_data_.channels()) {
        LOG(WARNING) << "Your dataset contains encoded images with mixed "
        << "channel sizes. Consider adding a 'force_color' flag to the "
        << "model definition, or rebuild your dataset using "
        << "convert_imageset.";
      }
    }
    decod_time += timer.MicroSeconds();

    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    const int offset_data = batch->data_.offset(item_id);
    const int offset_label = batch->label_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset_data);
    this->transformed_label_.set_cpu_data(top_label + offset_label);
    if (datum.encoded()) {
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    } else {
      this->data_transformer_->Transform_nv(datum, 
        &(this->transformed_data_),
        &(this->transformed_label_), cnt);
      ++cnt;
    }
    // if (this->output_labels_) {
    //   top_label[item_id] = datum.label();
    // }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  batch_timer.Stop();

#ifdef BENCHMARK_DATA
  LOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "  Dequeue time: " << deque_time / 1000 << " ms.";
  LOG(INFO) << "   Decode time: " << decod_time / 1000 << " ms.";
  LOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
#endif
}

INSTANTIATE_CLASS(CPMDataLayer);
REGISTER_LAYER_CLASS(CPMData);

}  // namespace caffe