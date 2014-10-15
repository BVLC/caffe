#include <opencv2/opencv.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>

#include <turbojpeg.h>

#include <google/protobuf/descriptor.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/download_manager.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#if CV_VERSION_MAJOR == 3
const int CV_LOAD_IMAGE_COLOR = cv::IMREAD_COLOR;
#endif

namespace caffe {
using boost::posix_time::ptime;
using boost::posix_time::microsec_clock;

using std::ofstream;

using google::protobuf::RepeatedPtrField;
using google::protobuf::Reflection;
using google::protobuf::FieldDescriptor;
using google::protobuf::Descriptor;
using google::protobuf::Message;

template <typename Dtype>
ProtobufDataLayer<Dtype>::~ProtobufDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  tjDestroy(jpeg_decompressor_);
}

template <typename Dtype>
void ProtobufDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_.mutable_cpu_data();
  if (this->output_labels_) {
    for (int label_index = 0; label_index < num_labels_; ++label_index) {
      this->prefetch_labels_.at(label_index)->mutable_cpu_data();
    }

    typedef typename vector<shared_ptr<Blob<Dtype> > >::iterator Iter;
    for (Iter iter = prefetch_weights_.begin(); iter != prefetch_weights_.end();
        ++iter) {
      (*iter)->mutable_cpu_data();
    }
  }
  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

// TODO(kmatzen): Remove this NOLINT.  Even after removing LayerSetUp,
// it still complained.
template <typename Dtype>
void ProtobufDataLayer<Dtype>::
    DataLayerSetUp(  // NOLINT(caffe/data_layer_setup)
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  jpeg_decompressor_ = tjInitDecompress();

  const ProtobufDataParameter& protobuf_data_param =
      this->layer_param_.protobuf_data_param();

  const int crop_height = protobuf_data_param.crop_height();
  const int crop_width  = protobuf_data_param.crop_width();
  CHECK((crop_height == 0 && crop_width == 0) ||
      (crop_height > 0 && crop_width > 0)) << "Current implementation requires "
      "crop_height and crop_width to be set at the same time.";

  const string& source = protobuf_data_param.source();
  LOG(INFO) << "Opening file " << source;
  ReadProtoFromBinaryFile(source, &manifest_);
  copy(manifest_.records().begin(), manifest_.records().end(),
       back_inserter(records_));

  if (protobuf_data_param.shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleProtobufs();
  }
  LOG(INFO) << "A total of " << records_.size() << " records.";

  for (int i = 0; i < manifest_.label_defs_size(); ++i) {
    const LabelDefinition& label_def = manifest_.label_defs(i);
    label_def_map_[label_def.name()] = label_def;
  }

  const RepeatedPtrField<string>& labels = protobuf_data_param.labels();
  int weight_index = 0;
  for (int label_index = 0; label_index < labels.size(); ++label_index) {
    const string& label = labels.Get(label_index);
    label_map_.insert(make_pair(label, label_index));

    if (label_def_map_.at(label).weights_size()) {
      weight_map_.insert(make_pair(label, weight_index++));
    }

    LOG(INFO) << "Selected label " << label;
  }

  CHECK_EQ(label_map_.size() + weight_map_.size() + 1, top.size());

  CHECK_LE(label_map_.size(), manifest_.label_defs_size());
  num_labels_ = label_map_.size();

  records_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (protobuf_data_param.rand_skip()) {
    unsigned int skip = caffe_rng_rand() % protobuf_data_param.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(records_.size(), skip) << "Not enough points to skip";
    records_id_ = skip;
  }
  // image
  const int batch_size = protobuf_data_param.batch_size();
  top.at(0)->Reshape(batch_size, 3, crop_height, crop_width);
  this->prefetch_data_.Reshape(batch_size, 3, crop_height, crop_width);
  LOG(INFO) << "output data size: " << top.at(0)->num() << ","
      << top.at(0)->channels() << "," << top.at(0)->height() << ","
      << top.at(0)->width();
  // label
  this->prefetch_labels_.resize(num_labels_);
  for (map<string, int>::const_iterator iter = label_map_.begin();
      iter != label_map_.end(); ++iter) {
    map<string, LabelDefinition>::const_iterator label_def_iter =
        label_def_map_.find(iter->first);
    CHECK(label_def_map_.end() != label_def_iter) << iter->first;
    const LabelDefinition& label_def = label_def_iter->second;
    CHECK(label_def.has_dim());
    const int& dims = label_def.dim();
    const int& label_index = iter->second;
    LOG(INFO) << "Label: " << iter->first << " " << label_index << " " << dims;
    top.at(label_index + 1)->Reshape(batch_size, dims, 1, 1);
    LOG(INFO) << "output " << iter->first
        << " size: " << top.at(label_index + 1)->num() << ","
        << top.at(label_index + 1)->channels() << ","
        << top.at(label_index + 1)->height() << ","
        << top.at(label_index + 1)->width();
    this->prefetch_labels_.at(label_index).reset(new Blob<Dtype>());
    this->prefetch_labels_.at(label_index)->Reshape(batch_size, dims, 1, 1);

    if (label_def.weights_size()) {
      prefetch_weights_.resize(prefetch_weights_.size() + 1);
      shared_ptr<Blob<Dtype> >& weights = prefetch_weights_.back();
      weights.reset(new Blob<Dtype>());
      weights->Reshape(batch_size, 1, 1, 1);
      LOG(INFO) << "create " << weights->count();

      top.at(prefetch_labels_.size() + prefetch_weights_.size())->Reshape(
          batch_size, 1, 1, 1);
    }
  }
}

template <typename Dtype>
void ProtobufDataLayer<Dtype>::ShuffleProtobufs() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(records_.begin(), records_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ProtobufDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  const ProtobufDataParameter& protobuf_data_param =
      this->layer_param_.protobuf_data_param();
  const int batch_size = protobuf_data_param.batch_size();
  const int crop_height = protobuf_data_param.crop_height();
  const int crop_width = protobuf_data_param.crop_width();

#ifdef DEBUG_PROTO_DATA
  const char* kDebugWindow = "protobuf debug";
  cv::namedWindow(kDebugWindow, cv::WINDOW_AUTOSIZE);
#endif

  boost::uniform_int<> distribution(-1.0f, 1.0f);
  boost::variate_generator<rng_t, boost::uniform_int<> >
      pixel_prng(*caffe_rng(), distribution);

  shared_ptr<DownloadManager> download_manager(download_manager_factory_());

  size_t records_size = records_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    int local_record_id = (records_id_ + item_id) % records_size;

    // get a record
    CHECK_GT(records_size, local_record_id);
    const ProtobufRecord& record = records_.at(local_record_id);

    CHECK(record.has_image_url());

    const string& image_url = record.image_url();

    download_manager->AddUrl(image_url);
  }

  download_manager->Download();

  const vector<shared_ptr<stringstream> >& jpeg_streams =
      download_manager->RetrieveResults();

  // datum scales
  DLOG(INFO) << "START";
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    int local_record_id = (records_id_ + item_id) % records_size;

    // get a record
    CHECK_GT(records_size, local_record_id);
    const ProtobufRecord& record = records_.at(local_record_id);
    boost::shared_ptr<stringstream> stream = jpeg_streams.at(item_id);

    const string& image_data_string = stream->str();
    vector<unsigned char> image_data_vector(image_data_string.begin(),
                                            image_data_string.end());
    if (image_data_string.empty()) {
      LOG(ERROR) << "Empty image " << record.image_url();
      continue;
    }

#ifdef NO_TURBO_JPEG
    cv::Mat image_data_mat(1, image_data_vector.size(), CV_8UC3,
                           image_data_vector.data());
    cv::Mat image = cv::imdecode(image_data_mat, CV_LOAD_IMAGE_COLOR);
    if (image.empty()) {
      LOG(ERROR) << "Failed to load " << record.image_url();
      continue;
    }
#else
    int jpeg_subsamp, image_width, image_height, jpeg_result;
    tjDecompressHeader2(jpeg_decompressor_, image_data_vector.data(),
                        image_data_vector.size(), &image_width, &image_height,
                        &jpeg_subsamp);
    cv::Mat image;
    try {
      image.create(image_height, image_width, CV_8UC3);
    } catch (const cv::Exception & ex) {
      LOG(ERROR) << ex.what() << " " << record.image_url()
          << " " << image_height << " " << image_width;
      continue;
    }
    jpeg_result = tjDecompress2(jpeg_decompressor_, image_data_vector.data(),
                                image_data_vector.size(), image.ptr(),
                                image_width, 0, image_height, TJPF_BGR,
                                TJFLAG_FASTDCT);
    if (jpeg_result) {
      LOG(ERROR) << "Failed to load " << record.image_url();
      continue;
    }
#endif

    CHECK(record.has_crop_x1());
    CHECK(record.has_crop_x2());
    CHECK(record.has_crop_y1());
    CHECK(record.has_crop_y2());

    int height = record.crop_y2() - record.crop_y1() + 1;
    int width = record.crop_x2() - record.crop_x1() + 1;

    cv::Mat crop(height, width, CV_8UC4);
    uchar* dst = crop.ptr();
    for (int row = 0; row < height; ++row) {
      const int src_row = record.crop_y1() + row;
      uchar* src_scanline = image.ptr(src_row);
      for (int col = 0; col < width; ++col) {
        const int src_col = record.crop_x1() + col;
        if (src_row >= 0 && src_row < image.rows &&
            src_col >= 0 && src_col < image.cols) {
          uchar* src = src_scanline + src_col * 3;
          for (int c = 0; c < 3; ++c) {
            *dst++ = *src++;
          }
          *dst++ = 255;
        } else {
          for (int c = 0; c < 4; ++c) {
            *dst++ = 0;
          }
        }
      }
    }

    cv::Mat resized_image;
    cv::Size size(crop_width, crop_height);
    cv::resize(crop, resized_image, size);

#ifdef DEBUG_PROTO_DATA
    LOG(INFO) << "DEBUG image " << image.size();
    imshow(kDebugWindow, image);
    cv::waitKey(0);

    LOG(INFO) << "DEBUG crop " << crop.size();
    imshow(kDebugWindow, crop);
    cv::waitKey(0);

    LOG(INFO) << "DEBUG resized " << resized_image.size();
    imshow(kDebugWindow, resized_image);
    cv::waitKey(0);
#endif
    Dtype* item_data = top_data + item_id * 3 * crop_height * crop_width;

    for (int row = 0; row < crop_height; ++row) {
      uchar* scanline = resized_image.ptr(row);
      for (int col = 0; col < crop_width; ++col) {
        uchar* pixel = scanline + col * 4;

        Dtype * pixel_base = item_data + row * crop_width + col;

        if (pixel[3] < 255) {
          // random
          for (int channel = 0; channel < 3; ++channel) {
            pixel_base[channel * crop_height * crop_width] = pixel_prng();
          }
        } else {
          // copy
          for (int channel = 0; channel < 3; ++channel) {
            pixel_base[channel * crop_height * crop_width] =
                (pixel[channel] - 128.0f) / 128.0f;
          }
        }
      }
    }

    const Reflection* record_reflection = record.GetReflection();
    CHECK_NOTNULL(record_reflection);
    for (map<string, int>::const_iterator iter = label_map_.begin();
        iter != label_map_.end(); ++iter) {
      const string& label_name = iter->first;
      const int& label_index = iter->second;
      Dtype* top_label = this->prefetch_labels_.at(label_index)
          ->mutable_cpu_data();

#ifdef WORKAROUND_PROTOBUF_FIND_EXTENSION
      vector<const FieldDescriptor*> fields;
      record_reflection->ListFields(record, &fields);
      const FieldDescriptor* extension_field = 0;
      for (int i = 0; i < fields.size(); ++i) {
        if (fields.at(i)->name() == "parent") {
          extension_field = fields.at(i);
        }
      }
#else
      const Descriptor* record_descriptor = record.GetDescriptor();
      CHECK_NOTNULL(record_descriptor);
      const FieldDescriptor* extension_field =
          record_descriptor->FindFieldByName("parent");
#endif
      CHECK_NOTNULL(extension_field);

      const Message& extension_message = record_reflection->GetMessage(record,
          extension_field);
      const Descriptor* descriptor = extension_message.GetDescriptor();
      CHECK_NOTNULL(descriptor);
      const FieldDescriptor* field = descriptor->FindFieldByName(label_name);
      CHECK(NULL != field) << label_name;
      const Reflection* extension_reflection =
          extension_message.GetReflection();
      CHECK_NOTNULL(extension_reflection);

      const LabelDefinition& label_def = label_def_map_.at(label_name);
      CHECK(label_def.has_dim());
      int label_dim = label_def.dim();
      Dtype* labels = top_label + item_id * label_dim;

      switch (field->label()) {
      case FieldDescriptor::LABEL_REPEATED: {
          for (int i = 0; i < label_dim; ++i) {
            switch (field->type()) {
            case FieldDescriptor::TYPE_UINT32:
              *labels = extension_reflection->GetRepeatedUInt32(
                  extension_message, field, i);
              break;
            case FieldDescriptor::TYPE_FLOAT:
              *labels = extension_reflection->GetRepeatedFloat(
                  extension_message, field, i);
              break;
            default:
              LOG(FATAL) << "label fields must be int32 or float";
              break;
            }
            if (label_def.has_mean()) {
              *labels -= label_def.mean();
            }
            if (label_def.has_stdev()) {
              CHECK(label_def.has_mean());
              *labels /= label_def.stdev();
            }
            ++labels;
          }
        }
        break;
      default: {
          CHECK_EQ(1, label_dim);
          switch (field->type()) {
          case FieldDescriptor::TYPE_UINT32:
            *labels = extension_reflection->GetUInt32(extension_message, field);
            break;
          case FieldDescriptor::TYPE_FLOAT:
            *labels = extension_reflection->GetFloat(extension_message, field);
            break;
          default:
            LOG(FATAL) << "label fields must be int32 or float";
            break;
          }
          if (label_def.has_mean()) {
            *labels -= label_def.mean();
          }
          if (label_def.has_stdev()) {
            CHECK(label_def.has_mean());
            *labels /= label_def.stdev();
          }
        }
        break;
      }

      if (label_def.weights_size()) {
        CHECK_EQ(1, label_dim);
        const Dtype* labels = top_label + item_id;

        int weight_index = weight_map_.at(label_name);
        Dtype* top_weight = this->prefetch_weights_.at(weight_index)
            ->mutable_cpu_data();

        Dtype* weights = top_weight + item_id;

        int label = *labels;

        CHECK_GT(label_def.weights_size(), label);

        float weight = label_def.weights(label);

        *weights = weight;
      }
    }
  }

#ifdef DEBUG_PREFETCH_DATA
  ofstream data("prefetch_data");
  data.write(reinterpret_cast<const char*>(this->prefetch_data_.cpu_data()),
      this->prefetch_data_.count() * sizeof(Dtype));
  data.close();

  for (int i = 0; i < num_labels_; ++i) {
    const boost::shared_ptr<Blob<Dtype> > labels = this->prefetch_labels_.at(i);
    stringstream filename;
    filename << "prefetch_label" << i;
    ofstream label(filename.str().c_str());
    label.write(reinterpret_cast<const char*>(labels->cpu_data()),
        labels->count() * sizeof(Dtype));
    label.close();
  }
  cv::waitKey(0);
  LOG(INFO) << "WAITING";
#endif

  // go to the next iter
  records_id_ += batch_size;
  if (records_id_ >= records_size) {
    // We have reached the end. Restart from the first.
    DLOG(INFO) << "Restarting data prefetching from start.";
    records_id_ = 0;
    if (protobuf_data_param.shuffle()) {
      ShuffleProtobufs();
    }
  }
  DLOG(INFO) << "END";
}

template <typename Dtype>
void ProtobufDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  ptime tic = microsec_clock::local_time();
  this->JoinPrefetchThread();
  ptime toc = microsec_clock::local_time();
  LOG(INFO) << "prefetch was behind by " << (toc - tic).total_milliseconds()
      << " milliseconds";
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             top.at(0)->mutable_cpu_data());
  if (this->output_labels_) {
    for (int i = 0; i < num_labels_; ++i) {
      caffe_copy(prefetch_labels_.at(i)->count(),
                 prefetch_labels_.at(i)->cpu_data(),
                 top.at(i + 1)->mutable_cpu_data());
    }

    for (int i = 0; i < prefetch_weights_.size(); ++i) {
      caffe_copy(prefetch_weights_.at(i)->count(),
          prefetch_weights_.at(i)->cpu_data(),
          top.at(i + num_labels_ + 1)->mutable_cpu_data());
    }
  }
  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(ProtobufDataLayer, Forward);
#endif

INSTANTIATE_CLASS(ProtobufDataLayer);
REGISTER_LAYER_CLASS(PROTOBUF_DATA, ProtobufDataLayer);

}  // namespace caffe
