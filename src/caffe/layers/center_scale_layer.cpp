// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/transfrom_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CenterScaleLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  TransformLayer<Dtype>::SetUp(bottom, top);
  
  // Initialize the mean/s
  
  has_mean_file_ = this->layer_param_.centerscale_param().has_mean_file();
  if (has_mean_file_) {
    const string& mean_file = this->layer_param_.centerscale_param().mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }

  int size_mean_value_ = this->layer_param_.centerscale_param().size_mean_value();
  if (size_mean_value_ > 0) {
    CHECK(!has_mean_file_) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int i = 0; i < size_mean_value_; ++i) {
      Dtype mean_value_ = this->layer_param_.centerscale_param().mean_value(i);
      mean_values_.pushback(mean_value_);
    }
  }

  // Initialize the scale/s
  int size_scale_value_ = this->layer_param_.centerscale_param().size_scale_value();
  if (size_scale_value_ > 0) {
    for (int i = 0; i < size_scale_value_; ++i) {
      Dtype scale_value_ = this->layer_param_.centerscale_param().scale_value(i);
      scale_values_.pushback(scale_value_);
    } 
  }

  // Initialize top blobs with the same size as bottom if not in-place
  for (int i = 1; i < bottom.size(); ++i) {
    if (bottom[i] != (*top)[i]) {
      (*top)[i]->ReshapeLike(bottom[i]);  
    }
  }
}

template <typename Dtype>
Dtype CenterScaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  for (int i = 0; i < bottom.size(); ++i) {
    int count = (*top)[i]->count();
    int num = (*top)[i]->num();
    int channels = (*top)[i]->channels();
    int dim = count / num;
    // Copy bottom_blob to top_blob
    Dtype* top_data = (*top)[i]->mutable_cpu_data();
    caffe_copy(count, bottom[i]->cpu_data(), top_data);
    if (has_mean_file_) {
      // Subtract the data_mean
      CHECK_EQ(count, data_mean_->count());
      caffe_apxy(count, Dtype(-1), data_mean_->cpu_data(), top_data);
    } else {
      switch (mean_values_->size()) {
      case 0:
        // Do nothing
        break;
      case 1:
        // Subtract mean_value
        Dtype mean_value = mean_values_[0];
        caffe_add_scalar(count, -mean_value, top_data);
        break;
      default:
        CHECK_EQ(channels, mean_values_->size()) <<
          "The number of mean_value_ should match the number of channels";
        // Subtract mean_value per channel
        for (int c = 0; c < channels; ++c) {
          Dtype mean_value = mean_values_[c];
          for (int n = 0; n < num; ++n) {
            int offset = (*top)[i]->offset(n,c);
            caffe_add_scalar(dim, -mean_value, top_data + offset);
          }
        }
      }
    }

    // Scale the values
    switch (scale_values_->size()) {
    case 0:
      // Do nothing
      break;
    case 1:
      Dtype scale_value_ = scale_values_[0];
      caffe_scal(count, scale_value_, top_data);
      break;
    default:
      CHECK(channels, scale_values_->size()) <<
        "The number of scale_value should match the number of channels";
      // Scale every channel independently
      for (int c = 0; c < channels; ++c) {
        Dtype scale_value_ = scale_values_[c];
        for (int n = 0; n < num; ++n) {
          int offset = (*top)[i]->offset(n,c);
          caffe_scal(count, scale_value_, top_data + offset);
        }
      }
    }
  }

  return Dtype(0.);
}


INSTANTIATE_CLASS(CenterScaleLayer);

}  // namespace caffe
