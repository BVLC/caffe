#include "caffe/layers/continuation_indicator_layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {
    template <typename Dtype>
    void ContinuationIndicatorLayer<Dtype>::Forward_gpu(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        CHECK_EQ(top[0]->shape()[0], time_step_) << "1st dimension of top blob should be same with time step.";
        CHECK_EQ(top[0]->shape()[1], mini_batch_) << "2nd dimension of top blob should be same with batch size.";
        Dtype* top_data = top[0]->mutable_gpu_data();
        caffe_gpu_set(mini_batch_, Dtype(0), top_data);
        caffe_gpu_set(mini_batch_*(time_step_ - 1), Dtype(1), top_data + mini_batch_);
    }
    template <typename Dtype>
    void ContinuationIndicatorLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom)
    {}
INSTANTIATE_LAYER_GPU_FUNCS(ContinuationIndicatorLayer);
}
