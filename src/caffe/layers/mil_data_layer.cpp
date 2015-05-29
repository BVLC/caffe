#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
MILDataLayer<Dtype>::~MILDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void MILDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>& top) {
  // LayerSetUp runs through the image name file and stores them.

  string label_file = this->layer_param_.mil_data_param().label_file().c_str();
  
  LOG(INFO) << "Loading labels from: "<< label_file;
  label_file_id_ = H5Fopen(label_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT); 

  LOG(INFO) << "MIL Data layer:" << std::endl;
  std::ifstream infile(this->layer_param_.mil_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.mil_data_param().source() << std::endl;

  const int channels = this->layer_param_.mil_data_param().channels();
  const int img_size = this->transform_param_.crop_size();
  const int scale = this->transform_param_.scale();
  const int images_per_batch = this->layer_param_.mil_data_param().images_per_batch();
  const int n_classes = this->layer_param_.mil_data_param().n_classes();
  const int num_scales = this->layer_param_.mil_data_param().num_scales();
  const float scale_factor = this->layer_param_.mil_data_param().scale_factor();
  mean_value_.clear();
  std::copy(this->transform_param_.mean_value().begin(),
      this->transform_param_.mean_value().end(),
      std::back_inserter(mean_value_));
  
  if(mean_value_.size() == 0)
    for(int i = 0; i < channels; i++)
      mean_value_.push_back((float)128.0);

  CHECK_EQ(mean_value_.size(), channels);

  LOG(INFO) << "MIL Data Layer: "<< "channels: " << channels;
  LOG(INFO) << "MIL Data Layer: "<< "img_size: " << img_size;
  LOG(INFO) << "MIL Data Layer: "<< "scale: " << scale;
  LOG(INFO) << "MIL Data Layer: "<< "n_classes: " << n_classes;
  LOG(INFO) << "MIL Data Layer: "<< "num_scales: " << num_scales;
  LOG(INFO) << "MIL Data Layer: "<< "scale_factor: " << scale_factor;
  LOG(INFO) << "MIL Data Layer: "<< "images_per_batch: " << images_per_batch;
  for(int i = 0; i < mean_value_.size(); i++)
    LOG(INFO) << "MIL Data Layer: "<< "mean_value[" << i << "]: " << mean_value_[i];
  
  const bool prefetch_needs_rand =
      this->transform_param_.mirror() ||
      this->transform_param_.crop_size();
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }
  
  string im_name;
  int count = 0; 
  while (infile >> im_name) {
    string full_im_name = string(this->layer_param_.mil_data_param().root_dir() + "/" + im_name + "." + this->layer_param_.mil_data_param().ext());
    image_database_.push_back(std::make_pair(im_name, full_im_name));
    count = count+1;
    if(count % 1000 == 0)
      LOG(INFO) << "num: " << count << ", image_name: " << im_name;
  }
  num_images_ = count;
  LOG(INFO) << "Number of images: " << count;

  top[0]->Reshape(images_per_batch*num_scales, channels, img_size, img_size);
  this->prefetch_data_.Reshape(images_per_batch*num_scales, channels, img_size, img_size);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  
  // label
  top[1]->Reshape(images_per_batch, n_classes, 1, 1);
  this->prefetch_label_.Reshape(images_per_batch, n_classes, 1, 1);

  this->counter_ = 0;
}

template <typename Dtype>
unsigned int MILDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

// Thread fetching the data
template <typename Dtype>
void MILDataLayer<Dtype>::InternalThreadEntry() {
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  const int img_size = this->transform_param_.crop_size();
  const int channels = this->layer_param_.mil_data_param().channels();
  const int scale = this->transform_param_.scale();
  const bool mirror = this->transform_param_.mirror();

  const int images_per_batch = this->layer_param_.mil_data_param().images_per_batch();
  const int n_classes = this->layer_param_.mil_data_param().n_classes();
  const int num_scales = this->layer_param_.mil_data_param().num_scales();
  const float scale_factor = this->layer_param_.mil_data_param().scale_factor();

  // zero out batch
  caffe_set(this->prefetch_data_.count(), Dtype(0), top_data);
  int item_id;
  for(int i_image = 0; i_image < images_per_batch; i_image++){
    // Sample which image to read
    unsigned int index = counter_; counter_ = counter_ + 1;
    const unsigned int rand_index = this->PrefetchRand();
    if(this->layer_param_.mil_data_param().randomize())
      index = rand_index;

    // LOG(INFO) << index % this->num_images_ << ", " << this->num_images_;
    pair<std::string, std::string> p = this->image_database_[index % this->num_images_];
    string im_name = p.first;
    string full_im_name = p.second;
    
    cv::Mat cv_img = cv::imread(full_im_name, CV_LOAD_IMAGE_COLOR);
    if (!cv_img.data) {
      LOG(ERROR) << "Could not open or find file " << full_im_name;
      return;
    }
    
    // hdf5_load_nd_dataset(this->label_file_id_, string("/label-"+im_name).c_str(), 2, 2, label_blob_);
    hdf5_load_nd_dataset(this->label_file_id_, string("/labels-"+im_name).c_str(), 2, 2, &this->label_blob_);
    const Dtype* label = label_blob_.mutable_cpu_data();
    // LOG(INFO) << "[width, height, channels, num] = " << label_blob_.width() << 
    //  ", " << label_blob_.height() << ", " << label_blob_.channels() << ", " << label_blob_.num();
    CHECK_EQ(label_blob_.width(), 1) << "Expected width of label to be 1." ;
    CHECK_EQ(label_blob_.height(), n_classes) << "Expected height of label to be " << n_classes;
    
    float img_size_i = img_size;
    for(int i_scales = 0; i_scales < num_scales; i_scales++){
      // Resize such that the image is of size img_size, img_size
      item_id = i_image*num_scales + i_scales;
      // LOG(INFO) << "MIL Data Layer: scale: " << (int) round(img_size_i);
      cv::Mat cv_cropped_img = this->data_transformer_->Transform_IDL(cv_img, (int) round(img_size_i), mirror);
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < cv_cropped_img.rows; ++h) {
          for (int w = 0; w < cv_cropped_img.cols; ++w) {
            Dtype pixel =
                static_cast<Dtype>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]);
            top_data[((item_id * channels + c) * img_size + h)
                     * img_size + w]
                = (pixel - (Dtype)mean_value_[c])*scale;
          }
        }
      }
      img_size_i = std::max((float)1., img_size_i*scale_factor);
    }
      
    for(int i_label = 0; i_label < n_classes; i_label++){
      top_label[i_image*n_classes + i_label] = 
        label[i_label];
    }
  }
}

INSTANTIATE_CLASS(MILDataLayer);
REGISTER_LAYER_CLASS(MILData);

}  // namespace caffe
