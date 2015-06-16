#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

LayerParameter GetSPPParam(const ROIPoolingParameter roi_pooling_param) {
    LayerParameter spp_param;
    spp_param.mutable_spp_param()->set_pyramid_height(roi_pooling_param.pyramid_height());
    switch (roi_pooling_param.pool()) {
        case ROIPoolingParameter_PoolMethod_MAX:
            spp_param.mutable_spp_param()->set_pool(
                    SPPParameter_PoolMethod_MAX);
            break;
        case ROIPoolingParameter_PoolMethod_AVE:
            spp_param.mutable_spp_param()->set_pool(
                    SPPParameter_PoolMethod_AVE);
            break;
        case ROIPoolingParameter_PoolMethod_STOCHASTIC:
            spp_param.mutable_spp_param()->set_pool(
                    SPPParameter_PoolMethod_STOCHASTIC);
            break;
        default:
            LOG(FATAL) << "Unknown pooling method.";
    }

    return spp_param;
}

LayerParameter GetConcatParam(void) {
    LayerParameter concat_param;
    concat_param.mutable_concat_param()->set_axis(0);
    concat_param.mutable_concat_param()->set_concat_dim(0);

    return concat_param;
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    spp_top_vec_.clear();
    spp_bottom_vec_.clear();
    concat_bottom_vec_.clear();

    // SPP layer output holder setup
    split_top_ = new Blob<Dtype>();

    // SPP layer param
    ROIPoolingParameter roi_pooling_param = this->layer_param_.roi_pooling_param();
    LayerParameter spp_param = GetSPPParam(roi_pooling_param);

    // SPP layer, top and bottom setup
    spp_top_vec_.push_back(new Blob<Dtype>());
    spp_bottom_vec_.push_back(bottom[0]);
    spp_layer_.reset(new SPPLayer<Dtype>(spp_param));
    spp_layer_.SetUp(spp_bottom_vec_, spp_top_vec_);

    // number of rois 
    n_rois_ = bottom[1]->num();

    // Concat layer param
    LayerParameter concat_param = GetConcatParam();

    // Concat layer, top and bottom
    for(int i = 0; i < n_rois_; i++) concat_bottom_vec_.push_back(spp_top_vec_[0]);
    concat_layer_.reset(new ConcatLayer<Dtype>(concat_param));
    concat_layer_.SetUp(concat_bottom_vec_, top);
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, (num, channels, height, width)";
    n_rois_ = bottom[1]->num();
    
    spp_layer_->Reshape(spp_bottom_vec_, spp_top_vec_);
    concat_layer_->Reshape(concat_bottom_vec_, top);
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();

  for(int i = 0; i < num_rois; i++) {
      // crop the bottom to create a bottom for SPP

}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ROIPoolingLayer);
#endif

INSTANTIATE_CLASS(ROIPoolingLayer);
REGISTER_LAYER_CLASS(ROIPooling);

}  // namespace caffe
