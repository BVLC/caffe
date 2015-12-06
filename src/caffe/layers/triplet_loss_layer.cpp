#include <algorithm>
#include <algorithm>
#include <vector>

#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);

    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), 1); // a
    CHECK_EQ(bottom[0]->width(), 1);  // a
    CHECK_EQ(bottom[1]->height(), 1); // p
    CHECK_EQ(bottom[1]->width(), 1);  // p
    CHECK_EQ(bottom[2]->height(), 1); // n
    CHECK_EQ(bottom[2]->width(), 1);  // n
    // cpu/gpu
    diff_a_p_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
    diff_a_n_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
    diff_n_p_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
    diff_p_a_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
    loss_.Reshape(bottom[0]->num(), 1, 1, 1);
    // gpu only
    diff_sq_a_p_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
    diff_sq_a_n_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
    dist_sq_a_p_.Reshape(bottom[0]->num(), 1, 1, 1);
    dist_sq_a_n_.Reshape(bottom[0]->num(), 1, 1, 1);
    // vector of ones used to sum along channels
    summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
    for (int i = 0; i < bottom[0]->channels(); ++i)
        summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    caffe_sub(
        bottom[0]->count(),
        bottom[0]->cpu_data(),         // a
        bottom[1]->cpu_data(),         // p
        diff_a_p_.mutable_cpu_data()); // a-p
    caffe_sub(
        bottom[0]->count(),
        bottom[0]->cpu_data(),         // a
        bottom[2]->cpu_data(),         // n
        diff_a_n_.mutable_cpu_data()); // a-n

    const int channels = bottom[0]->channels();
    Dtype margin = this->layer_param_.triplet_loss_param().margin();

    Dtype loss(0.0);
    for (int i = 0; i < bottom[0]->num(); ++i) {
        loss_.mutable_cpu_data()[i] = std::max<Dtype>(
            caffe_cpu_dot(channels, diff_a_p_.cpu_data() + (i * channels), diff_a_p_.cpu_data() + (i * channels))
            - caffe_cpu_dot(channels, diff_a_n_.cpu_data() + (i * channels), diff_a_n_.cpu_data() + (i * channels))
            + margin, Dtype(0.0));
        loss += loss_.cpu_data()[i];
    }
    loss /= static_cast<Dtype>(bottom[0]->num());
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    caffe_sub(
        bottom[0]->count(),
        bottom[2]->cpu_data(),
        bottom[1]->cpu_data(),
        diff_n_p_.mutable_cpu_data());
    caffe_sub(
        bottom[0]->count(),
        bottom[1]->cpu_data(),
        bottom[0]->cpu_data(),
        diff_p_a_.mutable_cpu_data());
    Blob<Dtype>* const diffs_[] = { &diff_n_p_, &diff_p_a_, &diff_a_n_ };
    for (int i = 0; i < 3; ++i) {
        if (propagate_down[i]) {
            int num = bottom[i]->num();
            int channels = bottom[i]->channels();
            Dtype* bout = bottom[i]->mutable_cpu_diff();
            Dtype delta = top[0]->cpu_diff()[0] / static_cast<Dtype>(bottom[i]->num()) * 2.0;
            for (int j = 0; j < num; ++j) {
                if (loss_.cpu_data()[j] > Dtype(0.0)) {
                    caffe_cpu_axpby(
                        channels,
                        delta,
                        diffs_[i]->cpu_data() + (j * channels),
                        Dtype(0.0),
                        bout + (j * channels));
                }
                else {
                    caffe_set(channels, Dtype(0), bout + (j * channels));
                }
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);
}  // namespace caffe
