#include <opencv2/core/core.hpp>

#include <stdint.h>

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
DocDataLayer<Dtype>::~DocDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  while (!prefetch_labels_.empty()) {
    delete prefetch_labels_.back();
	prefetch_labels_.pop_back();
  }
}

template <typename Dtype>
Dtype DocDataLayer<Dtype>::GetLabelValue(DocumentDatum& doc, const std::string& label_name) {
  if (label_name == "country") {
    return doc.has_country() ? doc.country() : missing_value_;
  } else if (label_name == "language") {
    return doc.has_language() ? doc.language() : missing_value_;
  } else if (label_name == "decade") {
    return doc.has_decade() ? doc.decade() : missing_value_;
  } else if (label_name == "column_count") {
    return doc.has_column_count() ? doc.column_count() : missing_value_;
  } else if (label_name == "possible_records") {
    return doc.has_possible_records() ? doc.possible_records() : missing_value_;
  } else if (label_name == "actual_records") {
    return doc.has_actual_records() ? doc.actual_records() : missing_value_;
  } else if (label_name == "pages_per_image") {
    return doc.has_pages_per_image() ? doc.pages_per_image() : missing_value_;
  } else if (label_name == "docs_per_image") {
    return doc.has_docs_per_image() ? doc.docs_per_image() : missing_value_;
  } else if (label_name == "machine_text") {
    return doc.has_machine_text() ? doc.machine_text() : missing_value_;
  } else if (label_name == "hand_text") {
    return doc.has_hand_text() ? doc.hand_text() : missing_value_;
  } else if (label_name == "layout_category") {
    return doc.has_layout_category() ? doc.layout_category() : missing_value_;
  } else if (label_name == "layout_type") {
    return doc.has_layout_type() ? doc.layout_type() : missing_value_;
  } else if (label_name == "record_type_broad") {
    return doc.has_record_type_broad() ? doc.record_type_broad() : missing_value_;
  } else if (label_name == "record_type_fine") {
    return doc.has_record_type_fine() ? doc.record_type_fine() : missing_value_;
  } else if (label_name == "media_type") {
    return doc.has_media_type() ? doc.media_type() : missing_value_;
  } else if (label_name == "is_document") {
    return doc.has_is_document() ? doc.is_document() : missing_value_;
  } else if (label_name == "is_graphical") {
    return doc.has_is_graphical_document() ? doc.is_graphical_document() : missing_value_;
  } else if (label_name == "is_historical") {
    return doc.has_is_historical_document() ? doc.is_historical_document() : missing_value_;
  } else if (label_name == "is_textual") {
    return doc.has_is_textual_document() ? doc.is_textual_document() : missing_value_;
  } else if (label_name == "collection") {
    return doc.has_collection() ? doc.collection() : missing_value_;
  } else {
    CHECK(0) << "Unrecognized label_name: " << label_name;
  }
  return 0;
}

template <typename Dtype>
void DocDataLayer<Dtype>::SampleDB() {
  float rand = image_transformer_->RandFloat(0,1);
  float cum_prob = 0;
  int i;
  for (i = 0; i < probs_.size(); i++) {
    cum_prob += probs_[i];
	if (cum_prob >= rand) {
	  break;
    }
  }
  if (i == probs_.size()) {
    i--;
  }
  cur_index_ = i;
}

template <typename Dtype>
void DocDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CreateImageTransformer(this->layer_param_.image_transform_param());
  DocDataParameter doc_param = this->layer_param_.doc_data_param();
  num_labels_ = doc_param.label_names_size();
  missing_value_ = doc_param.missing_value();
 
  CHECK(doc_param.sources_size()) << "No source DBs specified";
  CHECK_EQ(top.size(), num_labels_ + 1) << "Must have a top blob for each type of label";


  // set up the input dbs
  for (int i = 0; i < doc_param.sources_size(); i++) {
    // Open the ith database
    shared_ptr<db::DB> db;
	shared_ptr<db::Cursor> cursor;
    db.reset(db::GetDB(doc_param.backend()));
    db->Open(doc_param.sources(i), db::READ);

	cursor.reset(db->NewCursor());
    // Check if we should randomly skip a few data points
    if (doc_param.rand_skip()) {
      unsigned int skip = caffe_rng_rand() %
                          doc_param.rand_skip();
      LOG(INFO) << "Skipping first " << skip << " data points.";
      while (skip-- > 0) {
        cursor->Next();
        if (!cursor->valid()) {
          DLOG(INFO) << "Restarting data prefetching from start.";
          cursor->SeekToFirst();
        }
      }
    }
	// Push the db handle, cursor, and weight of the ith db
	dbs_.push_back(db);
	cursors_.push_back(cursor);
	if (i < doc_param.weights_size()) {
	  probs_.push_back(doc_param.weights(i));
	} else {
      probs_.push_back(1.0f);
	}
  }
  SampleDB();

  // normalize probability weights
  float sum = 0;
  for (int i = 0; i < probs_.size(); i++) {
    sum += probs_[i];
  }
  for (int i = 0; i < probs_.size(); i++) {
    probs_[i] /= sum;
  }

  // Read a data point, to initialize the prefetch and top blobs.
  DocumentDatum doc;
  doc.ParseFromString(cursors_[cur_index_]->value());

  vector<int> in_shape;
  in_shape.push_back(1);
  in_shape.push_back(doc.image().channels());
  in_shape.push_back(doc.image().width());
  in_shape.push_back(doc.image().height());

  // Use data_transformer to infer the expected blob shape from datum.
  image_transformer_->SampleTransformParams(in_shape);
  vector<int> top_shape = image_transformer_->InferOutputShape(in_shape);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = doc_param.batch_size();
  this->prefetch_data_.Reshape(top_shape);
  top[0]->ReshapeLike(this->prefetch_data_);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // labels
  for (int i = 0; i < num_labels_; i++) {
    string label_name = doc_param.label_names(i);
	label_names_.push_back(label_name);

    vector<int> label_shape(1, doc_param.batch_size());
    top[i + 1]->Reshape(label_shape);
	prefetch_labels_.push_back(new Blob<Dtype>(label_shape));
  }
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DocDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  double label_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.doc_data_param().batch_size();
  DocumentDatum doc;
  doc.ParseFromString(cursors_[cur_index_]->value());

  vector<int> in_shape;
  in_shape.push_back(1);
  in_shape.push_back(doc.image().channels());
  in_shape.push_back(doc.image().width());
  in_shape.push_back(doc.image().height());
  // Use image_transformer to infer the expected blob shape from doc
  image_transformer_->SampleTransformParams(in_shape);
  vector<int> top_shape = image_transformer_->InferOutputShape(in_shape);
  this->transformed_data_.Reshape(top_shape);
  DLOG(INFO) << "Prefetch db: " << cur_index_ << " Shape: " << 
  	this->transformed_data_.shape_string() << " Doc id: " << doc.id();
  // Reshape prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  this->prefetch_data_.Reshape(top_shape);

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  timer.Start();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a datum
    DocumentDatum doc;
    doc.ParseFromString(cursors_[cur_index_]->value());
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)

	cv::Mat pretransform_img = ImageToCVMat(doc.image(), doc.image().channels() == 3);
	cv::Mat posttransform_img;
	image_transformer_->Transform(pretransform_img, posttransform_img);

    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    image_transformer_->CVMatToArray(posttransform_img, this->transformed_data_.mutable_cpu_data());
    trans_time += timer.MicroSeconds();
    timer.Start();
    // Copy labels
	for (int i = 0; i < num_labels_; i++) {
      top_label = prefetch_labels_[i]->mutable_cpu_data();
      top_label[item_id] = GetLabelValue(doc, label_names_[i]);
	}
	label_time += timer.MicroSeconds();
	timer.Start();
    // go to the next item.
    cursors_[cur_index_]->Next();
    if (!cursors_[cur_index_]->valid()) {
      DLOG(INFO) << "Restarting data prefetching from start on db: " << cur_index_;
      cursors_[cur_index_]->SeekToFirst();
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
  DLOG(INFO) << "    Label time: " << label_time / 1000 << " ms.";

  // Choose a db at random to pull from on the next batch
  SampleDB();
}

template <typename Dtype>
void DocDataLayer<Dtype>::CreateImageTransformer(ImageTransformationParameter param) {
  vector<ImageTransformer<Dtype>*>* transformers = new vector<ImageTransformer<Dtype>*>();
  for (int i = 0; i < param.params_size(); i++) {
    ProbImageTransformParameter prob_param = param.params(i);
    vector<ImageTransformer<Dtype>*>* prob_transformers = new vector<ImageTransformer<Dtype>*>();
	vector<float> weights;

    float weight;

	// Resize
	for (int j = 0; j < prob_param.resize_params_size(); j++) {
	  ResizeTransformParameter resize_param = prob_param.resize_params(j); 
	  if (j < prob_param.resize_prob_weights_size()) {
	    weight = prob_param.resize_prob_weights(j);
	  } else {
	    weight = 1;
	  }
	  ImageTransformer<Dtype>* transformer = new ResizeImageTransformer<Dtype>(resize_param);
	  prob_transformers->push_back(transformer);
	  weights.push_back(weight);
	}

    // Linear
	for (int j = 0; j < prob_param.linear_params_size(); j++) {
	  LinearTransformParameter resize_param = prob_param.linear_params(j); 
	  if (j < prob_param.linear_prob_weights_size()) {
	    weight = prob_param.linear_prob_weights(j);
	  } else {
	    weight = 1;
	  }
	  ImageTransformer<Dtype>* transformer = new LinearImageTransformer<Dtype>(resize_param);
	  prob_transformers->push_back(transformer);
	  weights.push_back(weight);
	}

    ImageTransformer<Dtype>* prob_transformer = new ProbImageTransformer<Dtype>(prob_transformers, weights);
	transformers->push_back(prob_transformer);
  }
  image_transformer_ = new SequenceImageTransformer<Dtype>(transformers);
}

template <typename Dtype>
void DocDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  this->JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Reshape to loaded data.
  top[0]->ReshapeLike(this->prefetch_data_);
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  for (int i = 0; i < num_labels_; i++) {
    Blob<Dtype>* prefetch_label = prefetch_labels_[i];
    top[i + 1]->ReshapeLike(*prefetch_label);

    caffe_copy(prefetch_label->count(), prefetch_label->cpu_data(),
               top[i + 1]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  this->CreatePrefetchThread();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(DocDataLayer, Forward);
#endif

INSTANTIATE_CLASS(DocDataLayer);
REGISTER_LAYER_CLASS(DocData);

}  // namespace caffe
