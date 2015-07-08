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
}

template <typename Dtype>
void DocDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CreateImageTransformer(this->layer_param_.image_transform_param());
  // Initialize DB
  db_.reset(db::GetDB(this->layer_param_.data_param().backend()));
  db_->Open(this->layer_param_.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());

  // Check if we should randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      cursor_->Next();
    }
  }
  // Read a data point, to initialize the prefetch and top blobs.
  DocumentDatum doc;
  doc.ParseFromString(cursor_->value());

  vector<int> in_shape;
  in_shape.push_back(1);
  in_shape.push_back(doc.image().channels());
  in_shape.push_back(doc.image().width());
  in_shape.push_back(doc.image().height());

  // Use data_transformer to infer the expected blob shape from datum.
  this->image_transformer_->SampleTransformParams(in_shape);
  vector<int> top_shape = this->image_transformer_->InferOutputShape(in_shape);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = this->layer_param_.data_param().batch_size();
  this->prefetch_data_.Reshape(top_shape);
  top[0]->ReshapeLike(this->prefetch_data_);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, this->layer_param_.data_param().batch_size());
    top[1]->Reshape(label_shape);
    this->prefetch_label_.Reshape(label_shape);
  }
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DocDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  DocumentDatum doc;
  doc.ParseFromString(cursor_->value());

  vector<int> in_shape;
  in_shape.push_back(1);
  in_shape.push_back(doc.image().channels());
  in_shape.push_back(doc.image().width());
  in_shape.push_back(doc.image().height());
  // Use image_transformer to infer the expected blob shape from doc
  this->image_transformer_->SampleTransformParams(in_shape);
  vector<int> top_shape = this->image_transformer_->InferOutputShape(in_shape);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  this->prefetch_data_.Reshape(top_shape);

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  timer.Start();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a datum
    DocumentDatum doc;
    doc.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)

	cv::Mat pretransform_img = ImageToCVMat(doc.image(), doc.image().channels() == 3);
	cv::Mat posttransform_img;
	this->image_transformer_->Transform(pretransform_img, posttransform_img);

    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->image_transformer_->CVMatToArray(posttransform_img, this->transformed_data_.mutable_cpu_data());
    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = doc.layout_type();
    }
    trans_time += timer.MicroSeconds();
    timer.Start();
    // go to the next item.
    cursor_->Next();
    if (!cursor_->valid()) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      cursor_->SeekToFirst();
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
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

INSTANTIATE_CLASS(DocDataLayer);
REGISTER_LAYER_CLASS(DocData);

}  // namespace caffe
