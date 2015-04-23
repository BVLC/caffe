#include <boost/thread.hpp>
#include <opencv2/core/core.hpp>

#include <stdint.h>
#include <sys/socket.h>
#include <sys/stat.h>

#include <map>
#include <string>
#include <vector>

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
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    random_distribution_(),
    variate_generator_() {
  const DataParameter& data = param.data_param();
  if (data.probability_size()) {
    CHECK_EQ(data.source().size(), data.probability().size())
      << "Invalid DataParameter, there should be one probability per source";
    float sum = 0;
    for (int i = 0; i < data.probability().size(); ++i) {
      sum += data.probability(i);
    }
    CHECK_LT(fabsf(sum - 1.0f), 1e-6f)
      << "Invalid DataParameter, probabilities do not sum to 1";
  }
  for (int i = 0; i < data.source().size(); ++i) {
    URI uri(data.source(i));
    const shared_ptr<Scheme>& scheme(Scheme::get(uri.scheme()));
    readers_.push_back(scheme->get_reader(data, i));
  }
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Look at first data point to initialize the top blob.
  Datum* datum = readers_[0].get()->full().peek();

  bool force_color = this->layer_param_.data_param().force_encoded_color();
  if ((force_color && DecodeDatum(datum, true)) ||
      DecodeDatumNative(datum)) {
    LOG(INFO) << "Decoding Datum";
  }
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.data_param().batch_size();
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, datum->channels(), crop_size, crop_size);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, datum->channels(),
          crop_size, crop_size);
    }
    this->transformed_data_.Reshape(1, datum->channels(),
        crop_size, crop_size);
  } else {
    top[0]->Reshape(batch_size, datum->channels(),
        datum->height(), datum->width());
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, datum->channels(),
          datum->height(), datum->width());
    }
    this->transformed_data_.Reshape(1, datum->channels(),
        datum->height(), datum->width());
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
template <typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  const int batch_size = this->layer_param_.data_param().batch_size();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  bool force_color = this->layer_param_.data_param().force_encoded_color();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    Reader* reader = next_reader();
    const Datum& datum = *(reader->full().pop("Waiting on data reader"));

    // Reshape on single input batches for inputs of varying dimension.
    if (batch_size == 1 && crop_size == 0) {
      batch->data_.Reshape(1, datum.channels(),
          datum.height(), datum.width());
      this->transformed_data_.Reshape(1, datum.channels(),
          datum.height(), datum.width());
    }

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
    read_time += timer.MicroSeconds();
    timer.Start();

    // Apply data transformations (mirror, scale, crop...)
    Dtype* top_data = batch->data_.mutable_cpu_data();
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    if (datum.encoded()) {
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    } else {
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
    }
    if (this->output_labels_) {
      batch->label_.mutable_cpu_data()[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();

    reader->free().push(const_cast<Datum*>(&datum));
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

// This function is called on prefetch thread
template <typename Dtype>
Reader* DataLayer<Dtype>::next_reader() {
  const DataParameter& data = this->layer_param().data_param();
  // Default case without probabilities, try to find a reader with
  // data ready, or return first one
  if (data.probability_size() == 0) {
    for (int i = 0; i < readers_.size(); ++i) {
      Reader* reader = readers_[i].get();
      if (!reader->full().empty()) {
        return reader;
      }
    }
  } else {
    // Create RNG on current thread if first run
    if (!variate_generator_) {
      variate_generator_.reset(
          new boost::variate_generator<rng_t*, boost::uniform_real<float> >(
              caffe_rng(), random_distribution_));
    }
    // Pick reader randomly with probability
    boost::variate_generator<rng_t*, boost::uniform_real<float> >& rng =
        *variate_generator_.get();
    float rand = rng();
    for (int i = 0; i < data.probability().size(); ++i) {
      rand -= data.probability(i);
      if (rand < 0) {
        return readers_[i].get();
      }
    }
  }
  // If no data ready, or rounding error on probabilities
  return readers_[0].get();
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
