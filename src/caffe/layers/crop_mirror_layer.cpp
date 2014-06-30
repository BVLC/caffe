// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/transfrom_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CropMirrorLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  TransformLayer<Dtype>::SetUp(bottom, top);
  
  crop_size_ = this->layer_param_.cropmirror_param().crop_size();
  mirror_ = this->layer_param_.cropmirror_param().mirror();

  CHECK_GE(crop_size_, 0) <<
    "crop_size_ should be >= 0";
 
  // Initialize with the top blobs.
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_GE(bottom[i]->height(), crop_size_);
    CHECK_GE(bottom[i]->width(), crop_size_);
    if (crop_size_ > 0) {
      (*top)[i]->Reshape(bottom[i]->num(), bottom[i]->channels(), crop_size_, crop_size_);
    } else {
      (*top)[i]->ReshapeLike(bottom[i]);
    }
  }
}

template <typename Dtype>
Dtype CropMirrorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  for (int i = 0; i < bottom.size(); ++i) {
    // crop_mirror each blob
    caffe_crop_mirror(bottom[i], crop_size_, mirror_, (*top)[i]);
  }

  return Dtype(0.);
}


INSTANTIATE_CLASS(CropMirrorLayer);

}  // namespace caffe
