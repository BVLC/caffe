// ------------------------------------------------------------------
// R-FCN
// Written by Yi Li
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/box_annotator_ohem_layer.hpp"
#include "caffe/proto/caffe.pb.h"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

  template <typename Dtype>
  void BoxAnnotatorOHEMLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    BoxAnnotatorOHEMParameter box_anno_param =
      this->layer_param_.box_annotator_ohem_param();
    roi_per_img_ = box_anno_param.roi_per_img();
    CHECK_GT(roi_per_img_, 0);
    ignore_label_ = box_anno_param.ignore_label();
  }

  template <typename Dtype>
  void BoxAnnotatorOHEMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    num_ = bottom[0]->num();
    CHECK_EQ(5, bottom[0]->channels());
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    spatial_dim_ = height_*width_;

    CHECK_EQ(bottom[1]->num(), num_);
    CHECK_EQ(bottom[1]->channels(), 1);
    CHECK_EQ(bottom[1]->height(), height_);
    CHECK_EQ(bottom[1]->width(), width_);

    CHECK_EQ(bottom[2]->num(), num_);
    CHECK_EQ(bottom[2]->channels(), 1);
    CHECK_EQ(bottom[2]->height(), height_);
    CHECK_EQ(bottom[2]->width(), width_);

    CHECK_EQ(bottom[3]->num(), num_);
    bbox_channels_ = bottom[3]->channels();
    CHECK_EQ(bottom[3]->height(), height_);
    CHECK_EQ(bottom[3]->width(), width_);

    // Labels for scoring
    top[0]->Reshape(num_, 1, height_, width_);
    // Loss weights for bbox regression
    top[1]->Reshape(num_, bbox_channels_, height_, width_);
  }

  template <typename Dtype>
  void BoxAnnotatorOHEMLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void BoxAnnotatorOHEMLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }


#ifdef CPU_ONLY
  STUB_GPU(BoxAnnotatorOHEMLayer);
#endif

  INSTANTIATE_CLASS(BoxAnnotatorOHEMLayer);
  REGISTER_LAYER_CLASS(BoxAnnotatorOHEM);

}  // namespace caffe
