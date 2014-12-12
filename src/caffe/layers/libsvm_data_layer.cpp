#include <boost/algorithm/string.hpp>  // NOLINT(legal/copyright)
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {


// Parse feature and label from string of a line in LIBSVM format input.
template <typename Dtype>
void read_feature_and_label_form_string_or_die(
    string line, int channels, Datum* feature, Dtype* label) {
  // buff
  vector<string> cells, indval;

  // init feature datum
  feature->set_channels(channels);
  feature->set_height(1);
  feature->set_width(1);
  feature->clear_data();
  feature->clear_float_data();
  feature->mutable_float_data()->Reserve(channels);
  for (int s = 0; s < channels; s++) {
    feature->add_float_data(0.f);
  }

  // split lines
  boost::split(cells, line, boost::is_any_of(" \t"));
  *label = boost::lexical_cast<Dtype>(cells[0]);

  // Parse sparse format features
  float* pfeat = feature->mutable_float_data()->mutable_data();
  int i = 0;
  BOOST_FOREACH(string cell, cells) {
    if (i++ == 0) { continue; }
    boost::split(indval, cell, boost::is_any_of(":"));
    CHECK_EQ(indval.size(), 2);
    unsigned int ind = boost::lexical_cast<unsigned int>(indval[0]);
    Dtype val = boost::lexical_cast<Dtype>(indval[1]);
    pfeat[ind] = val;
  }
  return;
}

// Read libsvm format input and store them into `data` and `labels`
// Note that the `(*data)[...].label()` is dummy. Labels will be stored
// into `labels` in order to use floating value.
template <typename Dtype>
void read_libsvm_data(
    const string& source, int channels,
    vector<shared_ptr<Datum> >* data, vector<Dtype>* labels) {

  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  // read datum for each line
  string line;
  while (std::getline(infile, line)) {
    // trim spaces
    boost::trim(line);
    // skip empty lines
    if (line.empty()) {
      continue;
    }
    shared_ptr<Datum> datum(new Datum());
    Dtype label;
    read_feature_and_label_form_string_or_die(
        line, channels, datum.get(), &label);
    data->push_back(datum);
    labels->push_back(label);
  }

  // Check
  CHECK_EQ(data->size(), labels->size());
  BOOST_FOREACH(Dtype l, *labels) {
    CHECK((l == 0) || (l == 1) || (l == -1)) <<
        "In the current implementation, "
        "labels must be in {-1, 0, 1}. Found " << l;
  }
}

template <typename Dtype>
LIBSVMDataLayer<Dtype>::~LIBSVMDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void LIBSVMDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Number of features
  unsigned int channels = this->layer_param_.libsvm_data_param().channels();

  // Read data from file. Features and labels will be stored in data_
  // and labels_ respectively.
  const string& source = this->layer_param_.libsvm_data_param().source();
  read_libsvm_data(source, channels, &data_, &labels_);

  // Init accessor
  access_order_.resize(data_.size());
  for (int s = 0; s < access_order_.size(); s++) { access_order_[s] = s; }

  if (this->layer_param_.libsvm_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleAccessOrder();
  }
  LOG(INFO) << "A total of " << data_.size() << " samples.";

  // Check transform parameter
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const bool mirror = this->layer_param_.transform_param().mirror();
  CHECK_EQ(crop_size, 0) << "crop_size cannot be set.";
  CHECK(!mirror) << "mirror cannot be set.";

  // feature
  const int batch_size = this->layer_param_.libsvm_data_param().batch_size();
  (*top)[0]->Reshape(batch_size, channels, 1, 1);
  this->prefetch_data_.Reshape(batch_size, channels, 1, 1);
  // label
  (*top)[1]->Reshape(batch_size, 1, 1, 1);
  this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
}


// Suffule is done by suffling accessor sequence.
template <typename Dtype>
void LIBSVMDataLayer<Dtype>::ShuffleAccessOrder() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(access_order_.begin(), access_order_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void LIBSVMDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  LIBSVMDataParameter \
      libsvm_data_param = this->layer_param_.libsvm_data_param();
  const int batch_size = libsvm_data_param.batch_size();

  // datum scales
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    unsigned int pos = access_order_[pos_];
    CHECK_GT(data_.size(), pos);
    Datum * datum_p = data_[pos].get();
    // Apply transformations (scale, mean...) to the data
    this->data_transformer_.Transform(item_id, *datum_p, this->mean_, top_data);

    top_label[item_id] = labels_[pos];
    // go to the next iter
    pos_++;
    if (pos_ >= data_.size()) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      pos_ = 0;
      if (this->layer_param_.libsvm_data_param().shuffle()) {
        ShuffleAccessOrder();
      }
    }
  }
}

INSTANTIATE_CLASS(LIBSVMDataLayer);

}  // namespace caffe
