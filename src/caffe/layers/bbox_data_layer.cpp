#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/bbox_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
BboxDataLayer<Dtype>::~BboxDataLayer<Dtype>() {
    this->StopInternalThread();
}

template <typename Dtype>
void BboxDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.bbox_data_param().new_height();
  const int new_width  = this->layer_param_.bbox_data_param().new_width();
  const bool is_color  = this->layer_param_.bbox_data_param().is_color();
  string root_folder   = this->layer_param_.bbox_data_param().root_folder();

  // Read the file with image file names and label file namespace
  const string& source = this->layer_param_.bbox_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos;
  while (std::getline(infile, line)) {
      pos = line.find_last_of(' ');
      lines_.push_back(std::make_pair(line.substr(0, pos), line.substr(pos+1)));
  }

  CHECK(!lines_.empty()) << "File is empty";

  if (this->layer_param_.bbox_data_param().shuffle()) {
      // randomly shuffle data
      LOG(INFO) << "Shuffling data";
      const unsigned int prefetch_rng_seed = caffe_rng_rand();
      prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
      ShuffleImages();
  } else {
      if(this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
         this->layer_param_.bbox_data_param().rand_skip() == 0) {
        LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
      }
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.bbox_data_param().rand_skip()) {
      unsigned int skip = caffe_rng_rand() %
          this->layer_param_.bbox_data_param().rand_skip();
      LOG(INFO) << "Skipping first " << skip << " data points.";
      CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
      lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Get the expected blob shape from a cv_image
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  const int batch_size = this->layer_param_.bbox_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // bbox
  vector<int> bbox_shape;
  bbox_shape.push_back(batch_size);
  bbox_shape.push_back(1);
  bbox_shape.push_back(1);
  bbox_shape.push_back(1);
  vector<single_object> bbox_;
  infer_bbox_shape(root_folder + lines_[lines_id_].second, bbox_);
  bbox_shape[0] = bbox_.size() * 5 + 1;
  top[1]->Reshape(bbox_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(bbox_shape);
  }
}

template <typename Dtype>
void BboxDataLayer<Dtype>::infer_bbox_shape(const string& filename, std::vector<single_object>& bbox_) {
    std::ifstream infile(filename.c_str());
    std::istream_iterator<int> bbox_begin(infile), bbox_end;
    std::vector<int> bbox(bbox_begin, bbox_end);
    bbox_.clear();
    for(int i = 0; i < bbox.size();) {
        single_object obj;
        obj.xmin = bbox[i];
        obj.ymin = bbox[i+1];
        obj.xmax = bbox[i+2];
        obj.ymax = bbox[i+3];
        obj.class_idx = bbox[i+4];

        i += 5;
        bbox_.push_back(obj);
    }
}

template <typename Dtype>
void BboxDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void BboxDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  BboxDataParameter bbox_data_param = this->layer_param_.bbox_data_param();
  const int batch_size = bbox_data_param.batch_size();
  const int new_height = bbox_data_param.new_height();
  const int new_width = bbox_data_param.new_width();
  const bool is_color = bbox_data_param.is_color();
  string root_folder = bbox_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  std::vector<std::vector<single_object> > batch_bboxs;

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    // get the label data
    vector<single_object> bbox_;
    infer_bbox_shape(root_folder + lines_[lines_id_].second, bbox_);
    batch_bboxs.push_back(bbox_);

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // we have reached the end, Restart from the start
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.bbox_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }

  // Reshape the label blob to accomodate all the labels
  int total_batch_objs = 0;
  for (int i = 0; i < batch_bboxs.size(); ++i) {
      total_batch_objs += (batch_bboxs[i].size() * 5);
  }
  vector<int> label_shape_;
  label_shape_.push_back(total_batch_objs + batch_bboxs.size());
  label_shape_.push_back(1);
  label_shape_.push_back(1);
  label_shape_.push_back(1);
  batch->label_.Reshape(label_shape_);
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  for (int i = 0, idx = 0; i < batch_bboxs.size(); ++i, ++idx) {
      prefetch_label[idx] = batch_bboxs[i].size();
      for (int j = 0; j < batch_bboxs[i].size(); ++j) {
          prefetch_label[++idx] = batch_bboxs[i][j].xmin;
          prefetch_label[++idx] = batch_bboxs[i][j].ymin;
          prefetch_label[++idx] = batch_bboxs[i][j].xmax;
          prefetch_label[++idx] = batch_bboxs[i][j].ymax;
          prefetch_label[++idx] = batch_bboxs[i][j].class_idx;
      }
  }

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(BboxDataLayer);
REGISTER_LAYER_CLASS(BboxData);

} // namespace caffe
#endif // USE_OPENCV
