#include <algorithm>
#include <vector>

#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void TripletLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const int count = bottom[0]->count();
    caffe_gpu_sub(
        count,
        bottom[0]->gpu_data(),            // a
        bottom[1]->gpu_data(),            // p
        diff_a_p_.mutable_gpu_data());    // a_i-p_i
    caffe_gpu_powx(
        count,
        diff_a_p_.mutable_gpu_data(),     // a_i-p_i
        Dtype(2),
        diff_sq_a_p_.mutable_gpu_data()); // (a_i-p_i)^2
    caffe_gpu_gemv(
        CblasNoTrans,
        bottom[0]->num(),
        bottom[0]->channels(),
        Dtype(1.0),
        diff_sq_a_p_.gpu_data(),          // (a_i-p_i)^2
        summer_vec_.gpu_data(),
        Dtype(0.0),
        dist_sq_a_p_.mutable_gpu_data()); // \Sum (a_i-p_i)^2
    caffe_gpu_sub(
        count,
        bottom[0]->gpu_data(),            // a
        bottom[2]->gpu_data(),            // n
        diff_a_n_.mutable_gpu_data());    // a_i-n_i
    caffe_gpu_powx(
        count,
        diff_a_n_.mutable_gpu_data(),     // a_i-n_i
        Dtype(2),
        diff_sq_a_n_.mutable_gpu_data()); // (a_i-n_i)^2
    caffe_gpu_gemv(
        CblasNoTrans,
        bottom[0]->num(),
        bottom[0]->channels(),
        Dtype(1.0),
        diff_sq_a_n_.gpu_data(),          // (a_i-n_i)^2
        summer_vec_.gpu_data(),
        Dtype(0.0),
        dist_sq_a_n_.mutable_gpu_data()); // \Sum (a_i-n_i)^2
    Dtype margin = this->layer_param_.triplet_loss_param().margin();
    Dtype loss(0.0);
    for (int i = 0; i < bottom[0]->num(); ++i) {
        loss_.mutable_cpu_data()[i] = std::max<Dtype>(
            dist_sq_a_p_.cpu_data()[i]
            - dist_sq_a_n_.cpu_data()[i]
            + margin, Dtype(0.0));
        loss += loss_.cpu_data()[i];
    }
    loss /= static_cast<Dtype>(bottom[0]->num());
    top[0]->mutable_cpu_data()[0] = loss;
}

template<typename Dtype>
__global__ void CLLBackward(const int count, const int channels,
                            const Dtype delta,
                            const Dtype* loss, const Dtype* diff,
                            Dtype* bottom_diff) {
    CUDA_KERNEL_LOOP(i, count) {
        int n = i / channels; // the num index
        if (loss[n] > Dtype(0.0)) {
            bottom_diff[i] = delta * diff[i];
        }
        else {
            bottom_diff[i] = 0;
        }
    }
}

template<typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    caffe_gpu_sub(
        bottom[0]->count(),
        bottom[2]->gpu_data(),         // n
        bottom[1]->gpu_data(),         // p
        diff_n_p_.mutable_gpu_data()); // n-p
    caffe_gpu_sub(
        bottom[0]->count(),
        bottom[1]->gpu_data(),         // p
        bottom[0]->gpu_data(),         // a
        diff_p_a_.mutable_gpu_data()); // p-a
    Blob<Dtype>* const diffs_[] = { &diff_n_p_, &diff_p_a_, &diff_a_n_ };
    for (int i = 0; i < 3; ++i) {
        if (propagate_down[i]) {
            const int count = bottom[i]->count();
            int channels = bottom[i]->channels();
            const Dtype delta = top[0]->cpu_diff()[0] / static_cast<Dtype>(bottom[i]->num()) * 2.0;
            // NOLINT_NEXT_LINE(whitespace/operators)
            CLLBackward<Dtype><< < CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
                count, channels, delta,
                loss_.gpu_data(),
                diffs_[i]->gpu_data(),
                bottom[i]->mutable_gpu_diff());
            CUDA_POST_KERNEL_CHECK;
        }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);
}  // namespace caffe
