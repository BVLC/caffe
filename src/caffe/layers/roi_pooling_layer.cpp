#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
LayerParameter ROIPoolingLayer<Dtype>::GetSPPParam(const ROIPoolingParameter roi_pooling_param) {
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

template <typename Dtype>
LayerParameter ROIPoolingLayer<Dtype>::GetConcatParam(void) {
    LayerParameter concat_param;

    // concat along the first axis 
    concat_param.mutable_concat_param()->set_axis(0);

    return concat_param;
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // General comments: We will use separate SPP layers for each ROI. 
    // To avoid resizing issues we require that the number of ROIs be constant
    // and specified as a parameter to the ROIPooling layer

    // Currently, this layer only supports a batch size of 1, for ease of backprop
    // (see Fast-RCNN paper, section on image-centric sampling for mini-batches)
    LOG(INFO) << "Started";
    CHECK_EQ(bottom[0]->shape(0), 1) << "Can only support a batch size of 1";

    spp_layers_.clear();
    spp_outputs_.clear();
    spp_top_vecs_.clear();
    spp_bottom_vecs_.clear();
    concat_bottom_vec_.clear();

    // SPP layer param
    ROIPoolingParameter roi_pooling_param = this->layer_param_.roi_pooling_param();
    LayerParameter spp_param = GetSPPParam(roi_pooling_param);

    // number of rois 
    n_rois_ = roi_pooling_param.n_rois();

    // SPP layers, tops and bottoms setup
    for(int i = 0; i < n_rois_; i++) {
        spp_bottom_vecs_.push_back(new vector<Blob<Dtype>*>);
        spp_outputs_.push_back(new Blob<Dtype>());
        spp_top_vecs_.push_back(new vector<Blob<Dtype>*>);

        spp_bottom_vecs_[i]->push_back(bottom[0]);
        spp_top_vecs_[i]->push_back(spp_outputs_[i]);

        spp_layers_.push_back(shared_ptr<SPPLayer<Dtype> > (new SPPLayer<Dtype>(spp_param)));
        spp_layers_[i]->SetUp(*spp_bottom_vecs_[i], *spp_top_vecs_[i]);
    }

    // Concat layer param
    LayerParameter concat_param = GetConcatParam();

    // Concat layer, top and bottom
    for(int i = 0; i < n_rois_; i++) concat_bottom_vec_.push_back(spp_outputs_[i]);
    concat_layer_.reset(new ConcatLayer<Dtype>(concat_param));
    concat_layer_->SetUp(concat_bottom_vec_, top);
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, (num, channels, height, width)";

    // check if the number of rois is consistent
    CHECK_EQ(bottom[1]->shape(0), n_rois_) << "No. of ROIs not " << n_rois_;

    // reshape layers
    for(int i = 0; i < n_rois_; i++) {
        spp_layers_[i]->Reshape(*spp_bottom_vecs_[i], *spp_top_vecs_[i]);
    }
    concat_layer_->Reshape(concat_bottom_vec_, top);
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    Blob<Dtype>* bottom_data = bottom[0];
    Blob<Dtype>* bottom_rois = bottom[1];

    // Number of ROIs
    CHECK_EQ(n_rois_, bottom_rois->shape(0)) << "n_rois_ is not correct";
    CHECK_EQ(n_rois_, concat_bottom_vec_.size()) << "concat layer has"
        << concat_bottom_vec_.size() << " bottoms (expected " << n_rois_ << ")";

    // iterate over ROIs and pass them to SPP
    for(int i = 0; i < n_rois_; i++) {
        // get the current roi and check dimensions
        int xmin = static_cast<int>(bottom_rois->data_at(i, 0, 0, 0));
        int ymin = static_cast<int>(bottom_rois->data_at(i, 0, 0, 1));
        int xmax = static_cast<int>(bottom_rois->data_at(i, 0, 0, 2));
        int ymax = static_cast<int>(bottom_rois->data_at(i, 0, 0, 3));

        CHECK_GE(xmax, xmin);
        CHECK_GE(ymax, ymin);
        CHECK_GE(xmin, 0);
        CHECK_GE(ymin, 0);
        CHECK_LE(xmax, bottom_data->shape(3)-1);
        CHECK_LE(ymax, bottom_data->shape(2)-1);

        // create a bottom for SPP
        Blob<Dtype>* spp_bottom = new Blob<Dtype>();
        spp_bottom->Reshape(1, bottom_data->shape(1), ymax-ymin+1, xmax-xmin+1);
        for(int c = 0; c < spp_bottom->shape(1); c++) {
            for(int y = 0; y < spp_bottom->shape(2); y++) {
                int s_offset = bottom_data->offset(0, c, y+ymin, xmin);
                int d_offset = spp_bottom->offset(0, c, y, 0);
                caffe_copy(xmax-xmin+1, bottom_data->cpu_data()+s_offset, spp_bottom->mutable_cpu_data()+d_offset);
            }
        }
        (*spp_bottom_vecs_[i])[0] = spp_bottom;

        // set up SPP layer with new bottom
        spp_layers_[i]->SetUp(*spp_bottom_vecs_[i], *spp_top_vecs_[i]);

        // forward pass through SPP
        spp_layers_[i]->Forward(*spp_bottom_vecs_[i], *spp_top_vecs_[i]);
    }

    // forward pass through the concat layer
    concat_layer_->Forward(concat_bottom_vec_, top);
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
        LOG(FATAL) << "ROIPooling Layer cannot backpropagate to label inputs.";
    }
    if (!propagate_down[0]) {
        return;
    }

    CHECK_EQ(n_rois_, bottom[1]->shape(0)) << "n_rois_ is not correct";

    // backprop through concat
    vector<bool> concat_propagate_down(n_rois_, true);
    concat_layer_->Backward(top, concat_propagate_down, concat_bottom_vec_);

    // iterate over ROIs and accumulate gradients
    Blob<Dtype>* bottom_data = bottom[0];
    Blob<Dtype>* bottom_rois = bottom[1];
    vector<bool> spp_propagate_down(1, true);
    caffe_set(bottom_data->count(), Dtype(0), bottom_data->mutable_cpu_diff()); // initialize layer bottom diff to 0
    for(int i = 0; i < n_rois_; i++) {
        // backward pass through SPP layer
        spp_layers_[i]->Backward(*spp_top_vecs_[i], spp_propagate_down, *spp_bottom_vecs_[i]);

        Blob<Dtype>* spp_bottom = (*spp_bottom_vecs_[i])[0];
        // get the current roi and check dimensions
        int xmin = static_cast<int>(bottom_rois->data_at(i, 0, 0, 0));
        int ymin = static_cast<int>(bottom_rois->data_at(i, 0, 0, 1));
        int xmax = static_cast<int>(bottom_rois->data_at(i, 0, 0, 2));
        int ymax = static_cast<int>(bottom_rois->data_at(i, 0, 0, 3));

        CHECK_GE(xmax, xmin);
        CHECK_GE(ymax, ymin);
        CHECK_GE(xmin, 0);
        CHECK_GE(ymin, 0);
        CHECK_LE(xmax, bottom_data->shape(3)-1);
        CHECK_LE(ymax, bottom_data->shape(2)-1);
        CHECK_EQ(spp_bottom->shape(2), ymax-ymin+1);
        CHECK_EQ(spp_bottom->shape(3), xmax-xmin+1);

        for(int c = 0; c < spp_bottom->shape(1); c++) {
            for(int y = 0; y < spp_bottom->shape(2); y++) {
                int d_offset = bottom_data->offset(0, c, y+ymin, xmin);
                int s_offset = spp_bottom->offset(0, c, y, 0);
                const Dtype *s = spp_bottom->cpu_diff() + s_offset;
                Dtype *d = bottom_data->mutable_cpu_diff() + d_offset;
                caffe_add(xmax-xmin+1, s, d, d);
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(ROIPoolingLayer);
#endif

INSTANTIATE_CLASS(ROIPoolingLayer);
REGISTER_LAYER_CLASS(ROIPooling);

}  // namespace caffe
