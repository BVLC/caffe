#include "caffe/layers/reorg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template<typename Dtype>
    void ReorgLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
        CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
                    "allow in-place computation.";
        ReorgParameter reorg_param = this->layer_param_.reorg_param();
        CHECK_EQ(reorg_param.has_stride(), true) << this->type() << " Layer needs stride param.";
        reverse_ = reorg_param.reverse();
        stride_ = reorg_param.stride();
        channels_ = bottom[0]->channels();
        height_ = bottom[0]->height();
        width_ = bottom[0]->width();
        batch_num_ = bottom[0]->num();

        diff_.Reshape(batch_num_, channels_, height_, width_);

        if (reverse_) {
            reorged_channels_ = channels_ / (stride_ * stride_);
            reorged_width_ = width_ * stride_;
            reorged_height_ = height_ * stride_;
        } else {
            reorged_channels_ = channels_ * stride_ * stride_;
            reorged_height_ = height_ / stride_;
            reorged_width_ = width_ / stride_;
        }
    }

    template<typename Dtype>
    void ReorgLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
        top[0]->Reshape(batch_num_, reorged_channels_,
                        reorged_height_, reorged_width_);
    }

    template<typename Dtype>
    void ReorgLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
        const Dtype *bottom_data = bottom[0]->cpu_data();
        Dtype *top_data = top[0]->mutable_cpu_data();
        reorg_cpu(bottom_data, width_, height_,
                  channels_, batch_num_, stride_, reverse_, top_data);
    }

    template<typename Dtype>
    void ReorgLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
        if(!propagate_down[0]){
            return;
        }
        //const Dtype *top_diff = top[0]->cpu_diff();
        const Dtype *top_diff = diff_.mutable_cpu_diff();
        Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
        reorg_cpu(top_diff, width_, height_,
                  channels_, batch_num_, stride_, !reverse_, bottom_diff);
    }
#ifdef CPU_ONLY
STUB_GPU(ReorgLayer);
#endif
    INSTANTIATE_CLASS(ReorgLayer);

    REGISTER_LAYER_CLASS(Reorg);

}  // namespace caffe
